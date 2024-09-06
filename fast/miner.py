# The MIT License (MIT)
# Copyright © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import os
import math
import time
import boto3
import torch
import wandb
import typer
import bittensor as bt
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from dataset import SubsetFineWebEdu2Loader
from transformers import AutoTokenizer

import utils

# Instantiate my S3 client.
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',
    aws_access_key_id = AWS_ACCESS_KEY_ID,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY
)

# Main function.
def main(
    bucket: str = 'decis',
    batch_size: int = 6, 
    learning_rate: float = 5e-7,  
    optimizer_beta1: float = 0.9, 
    optimizer_beta2: float = 0.95, 
    optimizer_weight_decay: float = 0.1,
    num_pages: int = 2, 
    device: str = 'cuda:1', 
    use_wandb: bool = False,
):
    # Build hparams.
    hparams = SimpleNamespace(
        bucket = bucket,
        batch_size = batch_size, 
        num_pages = num_pages, 
        learning_rate = learning_rate,
        optimizer_beta1 = optimizer_beta1,
        optimizer_beta2 = optimizer_beta2,
        optimizer_weight_decay = optimizer_weight_decay,
        device = device, 
        use_wandb = use_wandb,
    )
            
    # Load the head model state. Init the tokenizer and optimizer.
    def load_training_state():
        # Wait until there is a training state head.
        while not utils.head_exists( CLIENT, hparams.bucket ):
            print('Waiting for model head...')
            time.sleep(1)
            
        # Load the latest state.
        model, metadata = utils.load_model_head( CLIENT, hparams.bucket )
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( metadata.tokenizer, verbose=False, clean_up_tokenization_spaces=True )
        tokenizer.pad_token = tokenizer.eos_token    
        optimizer = optim.AdamW(
            model.parameters(),
            lr = hparams.learning_rate,  # Peak learning rate
            betas = ( hparams.optimizer_beta1, hparams.optimizer_beta2 ), # B1 and B2
            weight_decay = hparams.optimizer_weight_decay  # Weight decay
        )
        print (f'Loaded training state: { metadata.filename }')
        return model, metadata, tokenizer, optimizer
    
    # Load the current training state.
    model, metadata, tokenizer, optimizer = load_training_state()
    model.to( hparams.device )
    
    # Init weights and biases
    if hparams.use_wandb:
        wandb.init(project='bistro', name = hparams.name, config = hparams )

    # Remember delta for later removal.
    last_update = None
    while True:
        # try:
            
        # Check training state and optionally reload.
        latest_meta = utils.get_head_metadata( CLIENT, hparams.bucket )
        if latest_meta == None or latest_meta.model_hash != metadata.model_hash:
            # Reload the training state.
            model, metadata, tokenizer, optimizer = load_training_state()
            model.to( hparams.device )
            print ('Loaded new state.')

        # Load the next dataset pages.
        dataset = SubsetFineWebEdu2Loader( 
            batch_size = hparams.batch_size, 
            sequence_length = metadata.sequence_length,
            num_pages = hparams.num_pages, 
            tokenizer = tokenizer
        )
                        
        # Iterate over the batches from these pages training the model.
        model.train()
        for _, batch in enumerate(tqdm(dataset, desc="Processing batches", leave=True)):
            
            # Shift the input ids to create labels.
            input_ids = torch.tensor(batch, dtype=torch.long).to(hparams.device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = tokenizer.pad_token_id

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Accumulate the gradients.
            loss.backward()
            
            # Step the optimizer
            optimizer.step()
            optimizer.zero_grad()
            
            # Log to wandb.
            tqdm.write(f"Loss: {loss.item()}")
            if hparams.use_wandb:
                wandb.log({ "Loss": loss.item(), "Perplexity": math.exp(loss.item()) } )
                    
        # Upload the latest update.
        print ('Uploading new update.')
        utils.delete( last_update, CLIENT, hparams.bucket )
        last_update = utils.upload_update( model, metadata, CLIENT, hparams.bucket ) 
                        
        # # Handle keyboard interrupts, stops training gracefully.
        # except (KeyboardInterrupt, SystemExit):
        #     utils.delete( last_update, CLIENT, hparams.bucket )
        #     break
        
        # # Handle unknown exceptions, continue training after 5 seconds.
        # except Exception as e:
        #     print (f"Error: {e}")
        #     time.sleep(5)
        #     continue


# Main function.
if __name__ == "__main__":
    typer.run(main)