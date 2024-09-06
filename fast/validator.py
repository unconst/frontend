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
import math
import time
import json
import boto3
import torch
import wandb
import typer
import random
from tqdm import tqdm
from typing import List, Dict
from dotenv import dotenv_values
from types import SimpleNamespace
from dataset import SubsetFineWebEdu2Loader
from transformers import AutoTokenizer

# Local utils.
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

# Uploads a dictionary of losses to S3
def upload_results( 
        results: Dict[ str, List[ float ] ], 
        model_hash: str,
        bucket: str 
    ) -> SimpleNamespace:
    upload_time = time.time()
    filename = f'results-{int(upload_time)}-{model_hash}.json'
    metadata = { 'filename': filename }
    with io.StringIO() as results_buffer:  # Use StringIO here
        json.dump( results, results_buffer, ensure_ascii=False )
        results_buffer.seek(0)            
        CLIENT.upload_fileobj(
            io.BytesIO( results_buffer.getvalue().encode('utf-8')), 
            bucket, 
            filename,
            ExtraArgs={"Metadata": metadata}
        )
    metanamepace = SimpleNamespace( **metadata )
    return metanamepace

# Main function.
def main(
    name: str = 'validator',
    bucket: str = 'decis',
    batch_size: int = 6, 
    num_pages: int = 2, 
    device: str = 'cuda:1', 
    use_wandb: bool = False,
):
    # Build validator hparams.
    hparams = SimpleNamespace(
        name = name,
        bucket = bucket,
        batch_size = batch_size, 
        num_pages = num_pages, 
        device = device, 
        use_wandb = use_wandb,
    )
        
    # Load the head model state. Init the tokenizer and optimizer.
    def load_training_state():
        # Wait until there is a training state head.
        while not utils.head_exists( CLIENT, hparams.bucket ):
            print ('Waiting for model head...')
            time.sleep(1)
            
        # Load the latest state.
        model, metadata = utils.load_model_head( CLIENT, hparams.bucket )
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( metadata.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True )
        tokenizer.pad_token = tokenizer.eos_token    
        print (f'Loaded training state: { metadata.filename }')
        return model, metadata, tokenizer
    
    # Load the current training state.
    head_model, metadata, tokenizer = load_training_state()
    
    # Init weights and biases
    if hparams.use_wandb:
        wandb.init(project='bistro', name = hparams.name, config = hparams )

    # Remember delta for later removal.
    results: Dict[ str, List[ float] ] = {}
    last_results: SimpleNamespace = None
    while True:
        try:
            
            # Check training state and optionally reload.
            latest_meta = utils.get_head_metadata( CLIENT, hparams.bucket )
            if latest_meta == None or latest_meta.model_hash != metadata.model_hash:
                # Reload the training state.
                head_model, metadata, tokenizer = load_training_state()
                results: Dict[ str, List[ float] ] = {} # Reset results.
                
            # Load the next dataset pages.
            dataset = SubsetFineWebEdu2Loader( 
                batch_size = hparams.batch_size, 
                sequence_length = metadata.sequence_length,
                num_pages = hparams.num_pages, 
                tokenizer = tokenizer
            )
            
            # Get all updates from bucket.
            updates: List[ SimpleNamespace ] = utils.get_updates( CLIENT, hparams.bucket )
            
            # Iterate over update running eval.
            for update in random.sample( updates, len(updates) ):
                
                # Load the update.
                try:
                    model = utils.apply_update( head_model, update, CLIENT ).to( hparams.device )
                    print (f'Applied model update: {update.filename}')
                except Exception as e:
                    print (f"Error: {e}")
                    continue
                
                # Instantiate the loss history.
                if update.filename not in results:
                    results[ update.filename ] = []
                
                # Sample the dataset.
                dataset = SubsetFineWebEdu2Loader( 
                    batch_size = hparams.batch_size, 
                    sequence_length = metadata.sequence_length, 
                    num_pages = 1,
                    tokenizer = tokenizer
                )
                            
                # Iterate over the batches from these pages training the model.
                model.eval()
                losses = []
                for _, batch in enumerate(tqdm(dataset, desc="Processing batches", leave=True)):
                
                    # Shift the input ids to create labels.
                    input_ids = torch.tensor( batch, dtype=torch.long ).to( hparams.device )
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = tokenizer.pad_token_id

                    # Forward pass
                    outputs = model( input_ids=input_ids, labels=labels )
                    losses.append( outputs.loss.item() )
                    
                # Extend results
                results[ update.filename ].extend( losses )
                    
            # if last_results != None: 
            #     CLIENT.delete_object( Bucket = hparams.bucket, Key = last_results.filename )
            last_results = upload_results( 
                results, 
                metadata.model_hash,
                hparams.bucket
            )
            print (f'Uploaded results: {last_results}')
    
        # Handle keyboard interrupts, stops training gracefully.   
        except (KeyboardInterrupt, SystemExit):
            # if last_results != None: 
            #     CLIENT.delete_object( Bucket = hparams.bucket, Key = last_results.filename )
            break
        
        # Handle unknown exceptions, continue training after 5 seconds.
        except Exception as e:
            print (f"Error: {e}")
            time.sleep(5)
            continue


# Main function.
if __name__ == "__main__":
    typer.run(main)