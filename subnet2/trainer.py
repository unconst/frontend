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

    # Gets the model state last modified.
    def get_state_last_modified() -> int:
        response = client.head_object( Bucket = hparams.bucket, Key=f'model.pt')
        return int(response['LastModified'].timestamp())

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
        
    # Return deltas
    def get_models():
        response = client.list_objects_v2(Bucket = hparams.bucket)
        file_names = [content['Key'] for content in response.get('Contents', [])]
        
        delta_info_list = []
        for file_name in file_names:
            if file_name.startswith('miner-') and file_name.endswith('.pt'):
                try:
                    delta_info = types.SimpleNamespace( file_name=file_name )
                    delta_info_list.append(delta_info)
                except Exception as e:
                    pass
        return delta_info_list

    def clean_state():
        response = client.list_objects_v2(Bucket = hparams.bucket)
        deleted_files_count = 0
        if 'Contents' in response:
            for item in response['Contents']:
                client.delete_object(Bucket = hparams.bucket, Key=item['Key'])
                deleted_files_count += 1

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
                        
    # Init the tokenizer.
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('gpt2', verbose=False, clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token

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
    clean_state()
    upload_state( base_model )
    all_losses = {}
    last_best_loss = 100000 # infinity.
    epsilon = 0.02

    try:
        while True:        
                        
            all_models = get_models()
            if hparams.use_wandb:
                wandb.log( { "Models": len(all_models) } )
            for model_meta in random.sample(all_models, len(all_models)):
                
                # Try apply new delta.
                try:
                    model = load_model( model_meta.file_name )
                    model.to( hparams.device )
                except:
                    continue
                
                # Eval the model.
                dataset = SubsetFineWebEdu2Loader( 
                    batch_size = hparams.batch_size, 
                    sequence_length = hparams.sequence_length, 
                    num_pages = hparams.num_pages, 
                    tokenizer = tokenizer
                )
                
                # Iterate over the batches from these pages
                model_total_loss = 0
                for idx, batch in enumerate(dataset):

                    # Shift the input ids and create labels
                    input_ids = torch.tensor(batch, dtype=torch.long).to(hparams.device)
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = tokenizer.pad_token_id

                    # Forward pass checking loss.
                    outputs = model(input_ids=input_ids, labels=labels)
                    model_total_loss += outputs.loss.item()
                    
                # Normalize the loss.
                model_total_loss /= idx
                threshold = last_best_loss - last_best_loss * epsilon
                print ('Checked:', 'Name', model_meta.file_name, 'Loss', model_total_loss, 'Best', last_best_loss, 'Threshold', threshold)
                if hparams.use_wandb:
                    wandb.log( { "Threshold": threshold, 'Loss': model_total_loss, 'Best': last_best_loss } )
                    
                # Set the total loss into the delta.
                all_losses[model_meta.file_name] = model_total_loss
                
                # If the loss beats the total lost by epsilon
                if model_total_loss < threshold:
                    print ('New Best:', 'Name', model_meta.file_name, 'Loss', model_total_loss, 'Best', last_best_loss, 'Threshold', threshold)
                    
                    # Set the new base model.
                    base_model = copy.deepcopy(model).to('cpu')
                    
                    # Upload the new state.
                    upload_state( base_model )
                    
                    # Empty the total loss.
                    all_losses = {}
                    
                    # Update the last best
                    last_best_loss = model_total_loss
                    
                    # Increase the epsilon.
                    epsilon = 0.1
                    
                else:
                    # Decrease the epsilon.
                    epsilon *= 0.99
                    
                if hparams.use_wandb:
                    wandb.log( { "Epsilon": epsilon, 'Dif': model_total_loss - threshold } )
                    
    except KeyboardInterrupt:
        clean_state()
        sys.exit()

    finally:
        clean_state()

# Main function.
if __name__ == "__main__":
    typer.run(main)
