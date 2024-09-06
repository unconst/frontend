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

import os
import math
import time
import boto3
import torch
import wandb
import typer
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from dataset import SubsetFineWebEdu2Loader
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, List
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

# Loads results from a validator results file.
def load_results( filename: str, bucket: str ) -> Dict[ str, List[ float ]]:
    with io.BytesIO() as result_buffer:
        response = CLIENT.head_object( Bucket = bucket, Key = filename )
        CLIENT.download_fileobj( bucket, filename, result_buffer)
        result_buffer.seek(0)
        return json.load( result_buffer )
    
# Loads results from a validator results file.
def load_all_results( bucket: str ) -> Dict[ str, List[ float ]]: 
    all_results = []   
    response = client.list_objects_v2( Bucket = bucket )
    for content in response.get('Contents', []):
        filename = content['Key']
        if filename.startswith('results-') and filename.endswith('.json'):
            all_results.append( load_results( filename, bucket ) )
    return all_results    

def apply_update( model: torch.nn.Module, filename: str, bucket: str ) -> torch.nn.Module:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        client.download_fileobj( bucket, filename, temp_file)  # Download the update file from S3
        temp_file_path: str = temp_file.name  # Get the temporary file path
        updated_model = copy.deepcopy( model )  # Create a copy of the model
        updated_model.load_state_dict( torch.load( temp_file_path, weights_only=True ) )  # Load the update state into the copied model
    os.unlink(temp_file_path)  # Delete the temporary file
    return updated_model 

def upload_head( model: torch.nn.Module, bucket: str )
    model_state_dict = model.state_dict()
    metadata = {
        'filename': HEAD_MODEL_NAME,
        'upload_time': time.time(),
        'model_hash': hash_model( model ),
        'param_count': str(sum(p.numel() for p in model.parameters())),
        'model_class': get_full_class_name(model),
        'config': json.dumps(model.config.to_dict()) if hasattr(model, 'config') else None,
        **{k: v for k, v in metadata.items()}
    }
    # Upload the metadata
    with io.StringIO() as meta_buffer:  # Use StringIO here
        json.dump(metadata, meta_buffer, ensure_ascii=False)
        meta_buffer.seek(0)            
        client.upload_fileobj(io.BytesIO(meta_buffer.getvalue().encode('utf-8')), bucket, HEAD_META_NAME)

    # Upload the model.
    with io.BytesIO() as module_buffer:
        torch.save(model_state_dict, module_buffer)
        module_buffer.seek(0)
        
        # Upload to S3 with metadata
        client.upload_fileobj(
            module_buffer, 
            bucket, 
            HEAD_MODEL_NAME, 
        )
    # Create the metadata namespace.
    metanamespace = SimpleNamespace( **metadata )
    return metanamespace

# Main function.
def main(
    name: str = 'miner',
    bucket: str = 'decis',
    batch_size: int = 6, 
    sequence_length: int = 1024,
    learning_rate: float = 5e-7,  
    optimizer_beta1: float = 0.9, 
    optimizer_beta2: float = 0.95, 
    optimizer_weight_decay: float = 0.1,
    tokenizer_name: str = 'gpt2',
    num_pages: int = 2, 
    device: str = 'cuda:1', 
    use_wandb: bool = False,
):
    # Build hparams.
    hparams = SimpleNamespace(
        name = name,
        bucket = bucket,
        batch_size = batch_size, 
        sequence_length = sequence_length, 
        num_pages = num_pages, 
        tokenizer_name = tokenizer_name,
        learning_rate = learning_rate,
        optimizer_beta1 = optimizer_beta1,
        optimizer_beta2 = optimizer_beta2,
        optimizer_weight_decay = optimizer_weight_decay,
        device = device, 
        use_wandb = use_wandb,
    )
    
    # Init the model.
    configuration = GPT2Config( output_hidden_states = False, n_positions = hparams.sequence_length )
    model = GPT2LMHeadModel( config = configuration )
    metadata = utils.upload_head( model, CLIENT, hparams.bucket, metadata = {'sequence_length': hparams.sequence_length, 'tokenizer_name': hparams.tokenizer_name })
    
    # Init weights and biases
    if hparams.use_wandb:
        wandb.init(project='bistro', name = hparams.name, config = hparams )

    last_best_loss = float('inf')
    epsilon = 0.1
    while True:
        try:            
            # Get all results from validators.
            all_results: Dict[ str, SimpleNamespace ] = load_all_results( hparams.bucket )

            # Combine all the loss information from the results.
            combined_losses: Dict[ str, List[float] ] = {}
            for next_results in all_results:
                for filename in next_results.keys():
                    if filename not in combined_losses:
                        combined_losses[ filename ] = []
                    combined_losses[ filename ].extend( next_results[ filename ] )
                    
            # Find the key with lowest average loss
            best_filename = None
            lowest_avg_loss = float('inf')
            for filename, losses in combined_losses.items():
                avg_loss = sum(losses) / len(losses)
                if avg_loss < lowest_avg_loss:
                    lowest_avg_loss = avg_loss
                    best_filename = filename
            
            # If the loss beats the total loss with epsilon shifting.
            threshold = last_best_loss - last_best_loss * epsilon
            if lowest_avg_loss < threshold:                
                # Update the last best
                last_best_loss = lowest_avg_loss
                
                # Apply best update
                model = apply_update( model, best_filename, bucket )
                
                # Load the best model.
                base_model = utils.load_model( best_model_name )
                    
                # Upload the new state.
                # upload_state( base_model )
                                        
                # Increase the epsilon.
                epsilon = 0.1
                    
            # if hparams.use_wandb:
            #     wandb.log( { "Epsilon": epsilon } )
            
            # # Check training state and optionally reload.
            # latest_meta = utils.get_head_metadata( CLIENT, hparams.bucket )
            # if latest_meta == None or latest_meta.model_hash != metadata.model_hash:
            #     # Reload the training state.
            #     model, metadata, tokenizer, optimizer = utils.load_training_state()
            #     model.to( hparams.device )

            # # Load the next dataset pages.
            # dataset = SubsetFineWebEdu2Loader( 
            #     batch_size = hparams.batch_size, 
            #     sequence_length = metadata.sequence_length,
            #     num_pages = hparams.num_pages, 
            #     tokenizer = tokenizer
            # )
                            
            # # Iterate over the batches from these pages training the model.
            # model.train()
            # for _, batch in enumerate(tqdm(dataset, desc="Processing batches", leave=True)):
                
            #     # Shift the input ids to create labels.
            #     input_ids = torch.tensor(batch, dtype=torch.long).to(hparams.device)
            #     labels = input_ids.clone()
            #     labels[:, :-1] = input_ids[:, 1:]
            #     labels[:, -1] = tokenizer.pad_token_id

            #     # Forward pass
            #     outputs = model(input_ids=input_ids, labels=labels)
            #     loss = outputs.loss

            #     # Accumulate the gradients.
            #     loss.backward()
                
            #     # Step the optimizer
            #     optimizer.step()
            #     optimizer.zero_grad()
                
            #     # Log to wandb.
            #     tqdm.write(f"Loss: {loss.item()}")
            #     if hparams.use_wandb:
            #         wandb.log({ "Loss": loss.item(), "Perplexity": math.exp(loss.item()) } )
                        
            # # Upload the latest update.
            # last_update = upload_update( model, metadata, CLIENT, hparams.bucket ) 
                        
        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
            # utils.delete( metadata, CLIENT, hparams.bucket )
            break
        
        # Handle unknown exceptions, continue training after 5 seconds.
        except Exception as e:
            print (f"Error: {e}")
            time.sleep(5)
            continue


# Main function.
if __name__ == "__main__":
    typer.run(main)