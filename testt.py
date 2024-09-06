import torch
import torch.nn as nn

import os
import sys
import math
import torch
import wandb
import typer
import torch.optim as optim
from types import SimpleNamespace
from transformers import AutoTokenizer
from dataset import SubsetFineWebEdu2Loader
from transformers import GPT2Config, GPT2LMHeadModel, AdamW

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("gpt2", verbose=False)
tokenizer.pad_token = tokenizer.eos_token

# Assume you have a model and input_data
configuration = GPT2Config(output_hidden_states=False, n_positions=10)
model = GPT2LMHeadModel(config=configuration)

# Move model to the appropriate device
#device = args.device
#model.to(device)
# 
# AdamW optimizer with specified parameters
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0111,  # Peak learning rate
)

dataset = SubsetFineWebEdu2Loader(batch_size=1, sequence_length=10, num_pages=1, tokenizer=tokenizer)
for idx, batch in enumerate(dataset):

     # Shift the input ids and create labels
    input_ids = torch.tensor(batch, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = tokenizer.pad_token_id

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    # Retain gradients w.r.t to the logits
    logits = outputs.logits
    logits.retain_grad()

    # Backward pass
    loss.backward()
    
    # get the gradients from the logits.
    logits_grads = logits.grad.clone()
    
    # Zero the grads.
    optimizer.zero_grad()

    # Do the full backward using the logits grads and the inputs
    # Perform the full backward pass using the logits gradients and the inputs
    final_layer_output = model(input_ids=input_ids, labels=labels).logits
    avg_grad_final_layer = logits_grads

    # Backward pass using the retained gradients
    final_layer_output.backward(gradient=avg_grad_final_layer)
    break
    # final_layer_output.backward(gradient=avg_grad_final_layer)

    # # Now, model par
    # ameters have gradients that you can use in your optimizer
    # optimizer.step()