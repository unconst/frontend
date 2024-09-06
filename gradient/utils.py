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
import json
import time
import types
import boto3
import torch
import typer
import wandb
import hashlib
import random
import argparse
import tempfile
import importlib
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from dataset import SubsetFineWebEdu2Loader
from transformers import GPT2Config, GPT2LMHeadModel

HEAD_MODEL_NAME = 'model_head.pt'
HEAD_META_NAME = 'model_head.meta'

def get_full_class_name(obj):
    """ Returns the full class name, including the module path. """
    return f"{obj.__module__}.{obj.__class__.__name__}"

def dynamic_model_loader(module_name, class_name):
    """ Dynamically load a class from a module using its name. """
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def head_exists( client: boto3.client, bucket: str ) -> bool:
    """
    Checks if a file exists in the specified S3 bucket.

    Args:
        client (boto3.client): The boto3 S3 client.
        bucket (str): The name of the S3 bucket.
        key (str): The key (file name) to check for existence.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    try:
        client.head_object(Bucket = bucket, Key = HEAD_MODEL_NAME)
        return True
    
    except client.exceptions.NoSuchKey:
        return False
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def hash_model( module: torch.nn.Module ) -> str:
    """
    Generates a SHA-256 hash of the model's state dictionary.

    This function iterates through the model's state dictionary, concatenates the byte representation
    of each parameter, and then generates a SHA-256 hash of this concatenated byte string.

    Args:
        model (torch.nn.Module): The model to hash.

    Returns:
        str: The SHA-256 hash of the model's state dictionary.
    """
    # Extract the state dictionary from the module which contains all the parameters.
    module_state_dict = module.state_dict()
    
    # Concatenate all the model state values into a single byte string.
    concatenated_model_states_bytes = b''.join(
        [value.cpu().numpy().tobytes() for value in module_state_dict.values()]
    )
    
    # Generate a SHA-256 hash from the concatenated bytes.
    module_hash = hashlib.sha256(concatenated_model_states_bytes).hexdigest()
    return module_hash

def upload_head( 
        model: torch.nn.Module, 
        client: boto3.client, 
        bucket: str,
        metadata: dict = {},
    ):
    """
    Uploads the model state to S3 with optional metadata.
    
    Args:
        model: The model whose state dict is to be uploaded.
        metadata: A dictionary containing custom metadata (optional).
    """
    model_state_dict = model.state_dict()
    metadata = {
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
        
def get_head_metadata( client: boto3.client, bucket: str ) -> SimpleNamespace:
    """
    Continuously checks and returns the last modified timestamp of the 'model_head.pt' object in the specified S3 bucket.

    Args:
        client (boto3.client): The boto3 S3 client.
        bucket (str): The name of the S3 bucket.

    Returns:
        int: The last modified timestamp of the 'model_head.pt' object.
    """
    if not head_exists( client, bucket ):
        return None
    
    # Attempt to get the head object of 'model_head.pt' from the specified bucket
    response = client.head_object( Bucket = bucket, Key = HEAD_MODEL_NAME )
    
    # Create a SimpleNamespace to store metadata
    raw_metadata = response.get('Metadata', {})
    def convert_value(value):
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    processed_metadata = {k: convert_value(v) for k, v in raw_metadata.items()}
    
    # Get the file content and add it to the metadata
    file_content = client.get_object(Bucket=bucket, Key=HEAD_META_NAME)['Body'].read().decode('utf-8')
    file_metadata = json.loads(file_content)
    
    metadata = SimpleNamespace(
        last_modified = int(response['LastModified'].timestamp()),
        content_length = response.get('ContentLength'),
        **processed_metadata,
        **file_metadata
    )
    return metadata

def load_model_head( client: boto3.client, bucket: str, device: str = 'cpu' ) -> torch.nn.Module:
    """
    Loads the model from S3 using metadata to determine the architecture dynamically.
    
    Args:
        client: Boto3 client to interact with S3.
        bucket: The S3 bucket containing the model.
    
    Returns:
        The loaded model and its last modified time.
    """
    # Check if the model exists
    if not head_exists(client, bucket):
        return None

    # Get metadata from S3
    metadata = get_head_metadata(client, bucket)
    
    # Extract model class and config from metadata
    full_class_name = metadata.model_class
    model_config = json.loads( metadata.config )
    
    # Extract module and class name from the full class name
    module_name, class_name = full_class_name.rsplit('.', 1)
    
    # Dynamically load the model class
    ModelClass = dynamic_model_loader(module_name, class_name)
    
    # Instantiate the model using the config
    if model_config:
        model = ModelClass(config=ModelClass.config_class.from_dict(model_config))
    else:
        model = ModelClass()
        
    with tempfile.NamedTemporaryFile( delete = False ) as temp_file:
        client.download_fileobj( bucket, HEAD_MODEL_NAME, temp_file )            
        temp_file_path: str = temp_file.name
        new_model_state_dict = torch.load( temp_file_path, weights_only=True, map_location = torch.device( device ) )
        model.load_state_dict(new_model_state_dict)
    os.unlink(temp_file_path)
            
    return model, metadata


def upload_model( 
        model: torch.nn.Module,
        head_metadata: SimpleNamespace,
        client: boto3.client, 
        bucket: str 
    ) -> str:
    """
    Uploads the gradient (difference) between the base model and the delta model to S3.

    Args:
        gradient (Dict[str, torch.TEnsor]): The gradient.
        head_metadata (Dict): Metadata of the base model.
        client (boto3.client): Boto3 client to interact with S3.
        bucket (str): The S3 bucket to upload the gradient to.

    Returns:
        str: The filename of the uploaded gradient.
    """
    filename = f'delta-{time.time()}-{head_metadata.last_modified}-{head_metadata.model_hash}.pt'
    metadata = {
        'upload_time': str(time.time()),
        'base_last_modified': head_metadata.last_modified,
        'base_model_hash': head_metadata.model_hash,
    }
    with io.BytesIO() as delta_buffer:
        torch.save( delta, delta_buffer )
        delta_buffer.seek(0)
        client.upload_fileobj( 
            delta_buffer, 
            bucket, 
            filename,
            ExtraArgs={"Metadata": metadata}
        )
    return filename


def get_all_model_metas(
        client: boto3.client, 
        bucket: str,
    ) -> SimpleNamespace:
    response = client.list_objects_v2( Bucket = bucket )
    all_file_name = [content['Key'] for content in response.get('Contents', [])]
    all_model_metas = []
    for mode_file_name in file_names:
        if mode_file_name.startswith('delta-') and mode_file_name.endswith('.pt'):
            try:
                response = client.head_object( Bucket = bucket, Key = mode_file_name)
                model_metadata = types.SimpleNamespace( 
                    file_name = file_name,
                    last_modified = int(response['LastModified'].timestamp()),
                    content_length = response.get('ContentLength'),
                    metadata = {key: value for key, value in response['Metadata'].items()}
                )
                all_model_metas.append(model_metadata)
            except Exception as e:
                pass
    return delta_info_list


# def get_model(
#         client: boto3.client, 
#         bucket: str,
#     ) -> SimpleNamespace:
#     response = client.list_objects_v2(Bucket = hparams.bucket)
#     file_names = [content['Key'] for content in response.get('Contents', [])]
#     model_info = []
#     for file_name in file_names:
#         if file_name.startswith('delta-') and file_name.endswith('.pt'):
#             try:
#                 response = client.head_object( Bucket = bucket, Key = file_name)
#                 model_metadata = types.SimpleNamespace( 
#                     file_name=file_name 
#                 )
#                 delta_info_list.append(delta_info)
#             except Exception as e:
#                 pass
#     return delta_info_list


# def load_miner_model( model_meta: SimpleNamespace, device: str = 'cpu' ):
#     # Extract module and class name from the full class name
#     module_name, class_name = full_class_name.rsplit('.', 1)
    
#     # Dynamically load the model class
#     ModelClass = dynamic_model_loader(module_name, class_name)
    
#     # Instantiate the model using the config
#     if model_config:
#         model = ModelClass(config = ModelClass.config_class.from_dict(model_config))
#     else:
#         model = ModelClass()
        
#     # Load the model from S3.
#     with tempfile.NamedTemporaryFile( delete = False ) as temp_file:
#         client.download_fileobj( bucket, HEAD_MODEL_NAME, temp_file )            
#         temp_file_path: str = temp_file.name
#         new_model_state_dict = torch.load( temp_file_path, weights_only=True, map_location = torch.device( device ) )
#         model.load_state_dict(new_model_state_dict)
#     os.unlink(temp_file_path)
            
#     return model 


# # Clears all grad file names
# def delete_files( file_names: List[ str ], client: boto3.client, bucket: str ):
#     for file_name in file_names:
#         if file_name == None: continue
#         client.delete_object( Bucket = bucket, Key = file_name )
    
# # Uploads the losses data to S3.
# def upload_validator_losses( losses, name, last_modified ):
    
#     filename = f'losses-{name}-{last_modified}.json'
#     with io.StringIO() as losses_buffer:  # Use StringIO here
#         json.dump(losses, losses_buffer, ensure_ascii=False)
#         losses_buffer.seek(0)            
#         client.upload_fileobj(io.BytesIO(losses_buffer.getvalue().encode('utf-8')), hparams.bucket, filename)
#     return filename