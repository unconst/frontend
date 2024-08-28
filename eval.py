# MIT License
import math
import torch
from transformers import AutoTokenizer
from dataset import SubsetFineWebEdu2Loader
from transformers import GPT2LMHeadModel, GPT2Config

# Load the tokenizer
batch_size = 1
sequence_length = 1024
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("gpt2", verbose=False)
tokenizer.pad_token = tokenizer.eos_token
dataset = SubsetFineWebEdu2Loader(batch_size=batch_size, sequence_length = sequence_length, num_pages=1, tokenizer=tokenizer)

# Load the GPT-2 model from the same directory as this file
import os
configuration = GPT2Config(output_hidden_states=False)
model = GPT2LMHeadModel(config=configuration)
model.load_state_dict(torch.load(os.path.expanduser('/home/setup/reduct/fineweb_model.pt')))

# Move model to the appropriate device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:2"
model.to(device)

# Function to compute perplexity
from tqdm import tqdm
import torch.nn.functional as F
def compute_perplexity(model, dataset):
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataset), desc="Evaluating"):
            input_ids = batch[0]
            total_loss = 0.0
            total_tokens = 0
            print(input_ids)
            for i in tqdm(range(1, len(input_ids)-1)):
                input_ids_slice = torch.tensor( [input_ids[:i]], dtype=torch.long).to(device)
                outputs_slice = model(input_ids=input_ids_slice)
                logits = outputs_slice.logits[:, -1, :]  # logits for the last position
                target_label = torch.tensor( [input_ids[i+1]], dtype=torch.long).to(device)  # label for the next token (i+1)
                loss = F.cross_entropy(logits, target_label)
                total_loss += loss.item()
                total_tokens += 1
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            print(f"Perplexity: {perplexity:.4f}")
                    
# Compute and print perplexity
perplexity = compute_perplexity(model, dataset)
print(f"Perplexity: {perplexity:.2f}")