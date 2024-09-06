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

def main(
    name: str = 'miner',
    bucket: str = 'decis',
    aws_access_key_id: str = env_config.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key: str = env_config.get('AWS_SECRET_ACCESS_KEY'),
    batch_size: int = 6, 
    sequence_length: int = 2048,
    learning_rate: float = 5e-7,  
    optimizer_beta1: float = 0.9, 
    optimizer_beta2: float = 0.95, 
    optimizer_weight_decay: float = 0.1,
    num_pages: int = 2, 
    device: str = 'cuda:1', 
    use_wandb: bool = False,
    baseline: bool = False,
):
    hparams = SimpleNamespace(
        name = name,
        bucket = bucket,
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        batch_size = batch_size, 
        sequence_length = sequence_length,
        num_pages = num_pages, 
        learning_rate = learning_rate,
        optimizer_beta1 = optimizer_beta1,
        optimizer_beta2 = optimizer_beta2,
        optimizer_weight_decay = optimizer_weight_decay,
        device = device, 
        use_wandb = use_wandb,
        baseline = baseline,
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

    # Clears all grad file names
    def clean_state( file_names ):
        deleted_files_count = 0
        for file_name in file_names:
            if file_name == None: continue
            if file_name.startswith('miner-') and file_name.endswith('.pt'):
                client.delete_object(Bucket=hparams.bucket, Key=file_name)
                deleted_files_count += 1

    # Returns the model state.
    def load_state():
        # Download the model file
        while True:
            try:
                start_time = time.time()
                configuration = GPT2Config( output_hidden_states=False, n_positions = hparams.sequence_length )
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
    
    # Uploads the model state to S3.
    def upload_model( model, name ) -> str:
        start_time = time.time()
        miner_name = f'miner-{name}.pt'
        model_state_dict = model.state_dict()
        with io.BytesIO() as module_buffer:
            torch.save(model_state_dict, module_buffer)
            module_buffer.seek(0)
            client.upload_fileobj( module_buffer, hparams.bucket, miner_name )
        end_time = time.time()
        print (f'Uploaded model: {end_time - start_time} seconds.')
        return miner_name
    
    # Load my tokenizer.
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('gpt2', verbose=False, clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load my model
    if not hparams.baseline:
        model, base_model_last_modified = load_state()
    else:
        configuration = GPT2Config( output_hidden_states=False, n_positions = hparams.sequence_length )
        model = GPT2LMHeadModel( config=configuration )
    model = model.to( hparams.device )
    
    # Init an optimizer.
    optimizer = optim.AdamW(
        model.parameters(),
        lr = hparams.learning_rate,  # Peak learning rate
        betas = ( hparams.optimizer_beta1, hparams.optimizer_beta2 ), # B1 and B2
        weight_decay = hparams.optimizer_weight_decay  # Weight decay
    )

    # Set up wandb.
    if hparams.use_wandb:
        wandb.init(project='subnet', name = hparams.name, config = hparams )

    # Remember my file names
    my_file_names = []
    try:
        while True:
            
            # Optionally update the model.
            if not hparams.baseline:
                if get_state_last_modified() != base_model_last_modified and not hparams.baseline:
                    clean_state( my_file_names )
                    model, base_model_last_modified = load_state()
                    model.to( hparams.device )
                    # Init an optimizer with random param pertubations for training variance.
                    def random_perturbation(value, perturbation=0.25):
                        return value * (1 + random.uniform(-perturbation, perturbation))
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr = random_perturbation(hparams.learning_rate),  # Peak learning rate
                        betas = (
                            random_perturbation(hparams.optimizer_beta1), 
                            random_perturbation(hparams.optimizer_beta2)
                        ), # B1 and B2
                        weight_decay = random_perturbation(hparams.optimizer_weight_decay)  # Weight decay
                    )

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
                        
            # Upload gradients
            if not hparams.baseline:
                file_name = upload_model( model, hparams.name )
                my_file_names.append( file_name )

    except (KeyboardInterrupt, SystemExit):
        clean_state( my_file_names )
        sys.exit()
        
    except Exception as e:
        print(e)
        clean_state( my_file_names )

    finally:
        clean_state(my_file_names)


# Main function.
if __name__ == "__main__":
    typer.run(main)