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
import time
import types
import boto3
import torch
import typer
import argparse
import tempfile
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
env_config = {**dotenv_values(".env"), **os.environ}

def main(
    name: str = 'trainer',
    bucket: str = 'decis',
    aws_access_key_id: str = env_config.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key: str = env_config.get('AWS_SECRET_ACCESS_KEY'),
    grads_per_step: int = 5,
    learning_rate: float = 0.0001, 
    optimizer_beta1: float = 0.9, 
    optimizer_beta2: float = 0.95, 
    optimizer_weight_decay: float = 0.1,
    device: str = 'cuda:1', 
):
    # Create the hparams item.
    hparams = SimpleNamespace(
        name = name,
        bucket = bucket,
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        grads_per_step = grads_per_step,
        learning_rate = learning_rate,
        optimizer_beta1 = optimizer_beta1,
        optimizer_beta2 = optimizer_beta2,
        optimizer_weight_decay = optimizer_weight_decay,
        device = device, 
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
        
    # Return gradients
    def get_gradients():
        response = client.list_objects_v2(Bucket = hparams.bucket)
        file_names = [content['Key'] for content in response.get('Contents', [])]
        gradient_info_list = []
        for file_name in file_names:
            if file_name.startswith('miner-') and file_name.endswith('.pt'):
                try:
                    parts = file_name.split('-')
                    name = parts[1]
                    last_checkpoint_time = parts[3]
                    pages_str = '-'.join(parts[5:]).rsplit('.pt', 1)[0]
                    pages = eval(pages_str)
                    gradient_info = types.SimpleNamespace(
                        name=name,
                        last_modified=int(last_checkpoint_time),
                        pages=pages,
                        file_name=file_name
                    )
                    gradient_info_list.append(gradient_info)
                except Exception as e:
                    pass
        return gradient_info_list

    def clean_state():
        response = client.list_objects_v2(Bucket = hparams.bucket)
        deleted_files_count = 0
        if 'Contents' in response:
            for item in response['Contents']:
                client.delete_object(Bucket = hparams.bucket, Key=item['Key'])
                deleted_files_count += 1

    def apply_gradient( model, grad ) -> bool:
        try:
            # Download the gradient file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                client.download_fileobj( hparams.bucket, grad.file_name, temp_file )
                temp_file_path = temp_file.name
                gradient_dict = torch.load(temp_file_path, weights_only = True)
            os.unlink(temp_file_path)
        
            # Apply the gradients to the model parameters
            for name, param in model.named_parameters():
                if name in gradient_dict:
                    if param.grad is None:
                        param.grad = gradient_dict[name].to(model.device)
                    else:
                        param.grad += gradient_dict[name].to(model.device)
                        
            return True
        except Exception as e:
            print (f'Failed to attain gradient: {grad.file_name}')
            return False
                
                        
    # Init the tokenizer.
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('gpt2', verbose=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Init the model.
    configuration = GPT2Config( output_hidden_states = False, n_positions = 1024 )
    model = GPT2LMHeadModel( config = configuration )

    # Move model to the appropriate device
    device = hparams.device
    model.to(device)
    model.train()

    # Upload the model state to teh bucket.
    clean_state()
    upload_state( model )
    current_last_modified = get_state_last_modified()

    # Init optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr = hparams.learning_rate,  # Peak learning rate
        betas = ( hparams.optimizer_beta1, hparams.optimizer_beta2 ), # B1 and B2
        weight_decay = hparams.optimizer_weight_decay  # Weight decay
    )

    # Train the model.
    already_applied = set()
    num_applied = 0
    try:
        while True:        
            
            # Iterate over gradients available.
            time.sleep(0.5)
            for grad in get_gradients():
                
                # Throw out stale of already applied gradients.
                if grad.file_name in already_applied:
                    continue
                
                if grad.last_modified != current_last_modified:
                    continue
                
                # Try apply new gradient.
                success = apply_gradient( model, grad )
                already_applied.add( grad.file_name )
                
                # If successfully applied gradient.
                if success:
                    num_applied += 1
                    print (f"{num_applied}/{hparams.grads_per_step}")
                    
                    # Normalize gradients based on grad accumulations.
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad /= hparams.grads_per_step
                    
                    # Now apply the gradients with the optimizer
                    optimizer.step()
                    optimizer.zero_grad()
                
                    # Break gradient loop if 
                    if num_applied >= hparams.grads_per_step:
                        upload_state( model )
                        current_last_modified = get_state_last_modified()
                        num_applied = 0
                    
    except KeyboardInterrupt:
        clean_state()
        sys.exit()

    finally:
        clean_state()

# Main function.
if __name__ == "__main__":
    typer.run(main)



        
        
        # # Loop until we have applied grads_per_step.
        # num_applied = 0
            
        # # Run until we have accumulated enough gradients.
        # while num_applied < config.grads_per_step:
            
        #     # Get available gradients.
        #     all_grads = get_gradients()
        #     print (f'found: {len(all_grads)} gradients.')
            
        #     # Iterate over available and apply them.
        #     for grad in all_grads:
                
        #         # Throw out stale of already applied gradients.
        #         if grad.file_name in already_applied:
        #             print (f'threw out old already applied gradient')
        #             continue
        #         # if grad.last_modified != last_modified:
        #         #     print (f'threw out stale gradient {grad.last_modified} != {last_modified}')
        #         #     continue
                
        #         # Apply new gradient.
        #         apply_gradient( model, grad )
        #         num_applied += 1
        #         already_applied.add( grad.file_name )
        #         print (f'applied gradient {num_applied}/{config.grads_per_step}')
                
        #         # Now apply the gradients with the optimizer
        #         optimizer.step()
        #         optimizer.zero_grad()
        #         print (f'Stepped optimizer.')
                
        #         # Break gradient loop if 
        #         if num_applied >= config.grads_per_step:
        #             break
                
        #     # Waiting for more gradients.
        #     time.sleep(1)
        #     print ('waiting for more gradients.')
            
        # # # Average out gradients.
        # # for name, param in model.named_parameters():
        # #     if param.grad is not None:
        # #         param.grad += param.grad / num_applied
                
        # # # Now apply the gradients with the optimizer
        # # optimizer.step()
        # # optimizer.zero_grad()
        # # print (f'Stepped optimizer.')
        
        # # Clean all lingering gradients.
        # clean_state()
                
        # # Upload the new state to the bucket.
        # upload_state( model )
        # last_modified = get_state_last_modified()
        # print (f'Uploaded new state.')

