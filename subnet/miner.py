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

def main(
    name: str = 'miner',
    bucket: str = 'decis',
    aws_access_key_id: str = env_config.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key: str = env_config.get('AWS_SECRET_ACCESS_KEY'),
    batch_size: int = 10, 
    num_pages: int = 1, 
    device: str = 'cuda:1', 
    use_wandb: bool = False,
):
    hparams = SimpleNamespace(
        name = name,
        bucket = bucket,
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        batch_size = batch_size, 
        num_pages = num_pages, 
        device = device, 
        use_wandb = use_wandb,
    )
    print(hparams)
    
    client: boto3.client = boto3.client(
        's3',
        region_name='us-east-1',
        aws_access_key_id = hparams.aws_access_key_id,
        aws_secret_access_key = hparams.aws_secret_access_key
    )

    # Gets the model state last modified.
    def get_state_last_modified() -> int:
        while True:
            try:
                response = client.head_object( Bucket = hparams.bucket, Key=f'model.pt')
                return int(response['LastModified'].timestamp())
            except Exception as e:
                print ('Waiting for training state.')
                time.sleep(1)
                pass
        
    # Returns true if the model state has updated.
    def should_load_state( last_checkpoint_time: int ) -> bool:
        if last_checkpoint_time == None: return True
        return get_state_last_modified() > last_checkpoint_time

    # Clears all grad file names
    def clean_state( file_names ):
        deleted_files_count = 0
        for file_name in file_names:
            if file_name.startswith('miner-') and file_name.endswith('.pt'):
                client.delete_object(Bucket=hparams.bucket, Key=file_name)
                deleted_files_count += 1

    # Returns the model state.
    def load_state():
        # Download the model file
        while True:
            try:
                start_time = time.time()
                configuration = GPT2Config( output_hidden_states=False, n_positions = 1024 )
                model = GPT2LMHeadModel( config=configuration )
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    client.download_fileobj( hparams.bucket, f'model.pt', temp_file )            
                    temp_file_path: str = temp_file.name
                    new_model_state_dict = torch.load(temp_file_path, weights_only=True)
                    model.load_state_dict(new_model_state_dict)
                os.unlink(temp_file_path)
                end_time = time.time()
                return model, get_state_last_modified()
            except Exception as e:
                print (f'Waiting for training state: {e}')
                time.sleep(1)
                pass

    # Uploads my model gradients to the bucket.
    def upload_gradients( model, last_checkpoint_time, pages, name ):
        start_time = time.time()
        gradient_dict = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
        with io.BytesIO() as gradient_buffer:
            torch.save(gradient_dict, gradient_buffer)
            gradient_buffer.seek(0)
            file_name = f'miner-{name}-gradient-{last_checkpoint_time}-pages-{pages}.pt'
            client.upload_fileobj(gradient_buffer, hparams.bucket, file_name )
        end_time = time.time()
        print (f'Uploaded gradient: {end_time - start_time} seconds.')
        return file_name
    
    # Load my tokenizer.
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('gpt2', verbose=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Load my model
    model, last_checkpoint_time = load_state()
    model.to( hparams.device )

    # Set up wandb.
    if hparams.use_wandb:
        wandb.init(project='subnet', name = hparams.name, config = hparams )

    # Remember my file names
    my_file_names = []
    n_steps = 0
    n_uploads = 0
    n_tokens = 0
    n_batches = 0
    n_pages = 0
    try:
        while True:
            
            # Optionally update the model.
            if should_load_state( last_checkpoint_time ):
                clean_state( my_file_names )
                model, last_checkpoint_time = load_state()
                model.to(hparams.device)
                print ('loaded new model state.')
                if hparams.use_wandb:
                    wandb.log({ "n_steps": n_steps })
                n_steps += 1

            # Load dataset.
            dataset = SubsetFineWebEdu2Loader( 
                batch_size = hparams.batch_size, 
                sequence_length = 1024, 
                num_pages = hparams.num_pages, 
                tokenizer = tokenizer
            )
            n_pages += hparams.num_pages
            
            # Zero any previous gradients.
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
                
            # Iterate over the batches from these pages
            model.train()
            for idx, batch in enumerate(tqdm(dataset, desc="Processing batches", leave=True)):

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
                tqdm.write(f"Loss: {loss.item()}")
                
                # Log to wandb.
                n_batches += 1
                n_tokens += hparams.batch_size * 1024
                if hparams.use_wandb:
                    wandb.log({ "Loss": loss.item(), "Perplexity": math.exp(loss.item()), 'Tokens': n_tokens, 'Batches': n_batches, 'Pages': n_pages } )
                    
                # Break the loop if the state has changed.
                current_state: int = get_state_last_modified()
                if current_state != last_checkpoint_time:
                    break

            # only upload if the state is correct.
            if current_state == last_checkpoint_time:

                # Normalize gradients based on num batches for proper accumulation.
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= idx
                        
                # Upload gradients
                file_name = upload_gradients( model, last_checkpoint_time, dataset.pages, hparams.name )
                my_file_names.append( file_name )
                n_uploads += 1
                if hparams.use_wandb:
                    wandb.log({ "n_uploads": n_uploads })

                
    except (KeyboardInterrupt, SystemExit):
        clean_state( my_file_names )
        sys.exit()
        
    except Exception as e:
        clean_state( my_file_names )

    finally:
        clean_state(my_file_names)


# Main function.
if __name__ == "__main__":
    typer.run(main)