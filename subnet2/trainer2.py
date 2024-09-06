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
import json
import copy
import time
import types
import boto3
import torch
import typer
import wandb
import random
import argparse
import tempfile
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from dataset import SubsetFineWebEdu2Loader
from transformers import GPT2Config, GPT2LMHeadModel
env_config = {**dotenv_values(".env"), **os.environ}

def main(
    name: str = 'trainer',
    bucket: str = 'decis',
    aws_access_key_id: str = env_config.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key: str = env_config.get('AWS_SECRET_ACCESS_KEY'),
    device: str = 'cuda:1', 
    batch_size:int = 10, 
    sequence_length: int = 2048,
    num_pages: int = 3,
    use_wandb: bool = False,
):
    # Create the hparams item.
    hparams = SimpleNamespace(
        name = name,
        bucket = bucket,
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        device = device, 
        batch_size = batch_size,
        sequence_length = sequence_length,
        num_pages = num_pages,
        use_wandb = use_wandb,
    )
    print(hparams)

    # Create your S3 connection.
    client: boto3.client = boto3.client(
        's3',
        region_name = 'us-east-1',
        aws_access_key_id = hparams.aws_access_key_id,
        aws_secret_access_key = hparams.aws_secret_access_key
    )
    
    def download_losses():
        response = client.list_objects_v2(Bucket=hparams.bucket)
        file_names = [content['Key'] for content in response.get('Contents', []) if content['Key'].startswith('losses-') and content['Key'].endswith('.json')]
        
        losses_list = []
        for file_name in file_names:
            with io.BytesIO() as module_buffer:
                client.download_fileobj(hparams.bucket, file_name, module_buffer)
                module_buffer.seek(0)
                losses = json.load(module_buffer)
                name = file_name.split('-')[1]
                last_modified = file_name.split('-')[2].split('.')[0]
                losses_list.append(SimpleNamespace(name=name, losses=losses, last_modified=last_modified))
        return losses_list
    
    # Returns the model state.
    def load_model( model_name:str ):
        # Download the model file
        start_time = time.time()
        configuration = GPT2Config( output_hidden_states=False, n_positions = hparams.sequence_length )
        model = GPT2LMHeadModel( config=configuration )
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            client.download_fileobj( hparams.bucket, model_name, temp_file )            
            temp_file_path: str = temp_file.name
            new_model_state_dict = torch.load(temp_file_path, weights_only=True)
            model.load_state_dict(new_model_state_dict)
        os.unlink(temp_file_path)
        end_time = time.time()
        return model

    # Uploads the model state to S3.
    def upload_state( model ):
        start_time = time.time()
        model_state_dict = model.state_dict()
        with io.BytesIO() as module_buffer:
            torch.save(model_state_dict, module_buffer)
            module_buffer.seek(0)
            client.upload_fileobj( module_buffer, hparams.bucket, f'model.pt' )
        end_time = time.time()
        print (f'Uploaded model: {end_time - start_time} seconds.')
        
    def clean_state():
        response = client.list_objects_v2(Bucket = hparams.bucket)
        deleted_files_count = 0
        if 'Contents' in response:
            for item in response['Contents']:
                client.delete_object(Bucket = hparams.bucket, Key=item['Key'])
                deleted_files_count += 1
                        
    # Init the model.
    configuration = GPT2Config( output_hidden_states = False, n_positions = hparams.sequence_length )
    base_model = GPT2LMHeadModel( config = configuration )

    # Move model to the appropriate device
    device = hparams.device
    base_model.to('cpu')
    
    # Set up wandb.
    if hparams.use_wandb:
        wandb.init(project='subnet', name = hparams.name, config = hparams )

    # Upload the model state to teh bucket.
    # clean_state()
    # upload_state( base_model )
    last_best_loss = 100000 # infinity.
    epsilon = 0.1

    try:
        while True:        
            
            # Download the losses from the validators.
            losses_info = download_losses()
            print (losses_info)
            
            # Get loss info from all validators.  
            time.sleep(12) # simulate block.   
            epsilon *= 0.999
            
            # Iterate over losses and join losses_info[key].losses table from similar keys.
            combined_losses = {}
            for losses_for_validator in losses_info:
                for miner_name, losses in losses_for_validator.losses.items():
                    if miner_name not in combined_losses:
                        combined_losses[miner_name] = []
                    combined_losses[miner_name].extend(losses)
                                
            # Find the key with lowest average loss
            best_model_name = None
            lowest_avg_loss = float('inf')
            for model_name, losses in combined_losses.items():
                print ('losses', losses)
                avg_loss = sum(losses) / len(losses)
                if avg_loss < lowest_avg_loss:
                    lowest_avg_loss = avg_loss
                    best_model_name = model_name
            
            # If the loss beats the total loss with epsilon shifting.
            threshold = last_best_loss - last_best_loss * epsilon
            if lowest_avg_loss < threshold:                
                # Update the last best
                last_best_loss = lowest_avg_loss

                # Load the best model.
                base_model = load_model( best_model_name )
                    
                # Upload the new state.
                # upload_state( base_model )
                                        
                # Increase the epsilon.
                epsilon = 0.1
                    
            if hparams.use_wandb:
                wandb.log( { "Epsilon": epsilon } )
                    
    except KeyboardInterrupt:
        # clean_state()
        sys.exit()

    finally:
        pass
        # clean_state()

# Main function.
if __name__ == "__main__":
    typer.run(main)
