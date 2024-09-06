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

import io
import os
import sys
import copy
import math
import time
import boto3
import torch
import wandb
import typer
import tempfile
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from dataset import SubsetFineWebEdu2Loader
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel

env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')

def main(
    name: str = 'miner',
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
        name = name,
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
    
    # Build S3 connection.
    client: boto3.client = boto3.client(
        's3',
        region_name='us-east-1',
        aws_access_key_id = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY
    )
    
    # Wait until there is a training state head.
    while not head_exists( client, hparams.bucket ):
        print ('Waiting for model head...')
        time.sleep(1)
        
    # Download the model from the training bucket.
    model, current_meta = load_model_head( client, hparams.bucket )
    head_state = model.state_dict()
    
    # Init the tokenizer via the model meta
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( current_meta.tokenizer, verbose=False, clean_up_tokenization_spaces=True )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Init the optimizer.
    optimizer = optim.AdamW(
        model.parameters(),
        lr = hparams.learning_rate,  # Peak learning rate
        betas = ( hparams.optimizer_beta1, hparams.optimizer_beta2 ), # B1 and B2
        weight_decay = hparams.optimizer_weight_decay  # Weight decay
    )
    
    # Init weights and biases
    if hparams.use_wandb:
        wandb.init(project='gradient', name = hparams.name, config = hparams )

    # Remember delta for later removal.
    delta_filenames = []
    try:
        while True:
            
            # Optionally update the model by applying the latest delta.
            latest_meta = get_head_metadata( client, bucket )
            if latest_meta.model_hash != current_meta.model_hash:
                delta_meta = get_delta_metadata( latest_meta.previous_delta, client, bucket )
                apply_delta(
                    model, 
                    delta_meta,
                    client, 
                    bucket,
                ).to( hparams.device )
                current_meta = latest_meta
                head_state = model.state_dict()
                
            # Load dataset.
            dataset = SubsetFineWebEdu2Loader( 
                batch_size = hparams.batch_size, 
                sequence_length = hparams.sequence_length, 
                num_pages = hparams.num_pages, 
                tokenizer = tokenizer
            )
                            
            # Iterate over the batches from these pages
            model.train()
            for idx, batch in enumerate(tqdm(dataset, desc="Processing batches", leave=True)):
                
                # Sanity check.
                optimizer.zero_grad()

                # Shift the input ids and create labels
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
                        
            current_state_dict = model.state_dict()
            delta_state_dict = { k: current_state_dict[k] - head_state[k] for k in current_state_dict.keys() }
            delta_filename = upload_delta( 
                delta_state_dict,
                current_meta,
                client,
                bucket,
            )
            delta_filenames.append( delta_filename )

    except (KeyboardInterrupt, SystemExit):
        delete_files( delta_filenames )
        sys.exit()
        
    except Exception as e:
        print(e)
        delete_files( delta_filenames )

    finally:
        delete_files( delta_filenames )


# Main function.
if __name__ == "__main__":
    typer.run(main)