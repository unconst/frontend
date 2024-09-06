from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

# Load the tokenizer
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('gpt2', verbose=False)
tokenizer.pad_token = tokenizer.eos_token

# Define the configuration for the model
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,       # Set the vocabulary size from the tokenizer
    hidden_size=2040,       # Hidden size adjusted to match the parameter count
    num_hidden_layers=12,   # Number of layers adjusted to match the parameter count
    num_attention_heads=12, # Number of attention heads
    intermediate_size=6144  # Intermediate size in feedforward layers
)

# Instantiate the model with the defined configuration (untrained)
model = LlamaForCausalLM(config)

# Compute the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"The model has {num_params} parameters.")