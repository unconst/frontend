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

AVAILABLE_METHODS = ['baseline', 'seqcompress', 'batchcompress', 'topk', 'rank']

# Argument parser for hyperparameters
def main(model_name: str = 'gpt2', 
    run_name: str = None,
    project_name: str = 'fineweb-tuning',
    method: str = 'seqcompress',
    batch_size: int = 12, 
    sequence_length: int = 2048, 
    learning_rate: float = 5e-7, 
    device: str = 'cuda:1', 
    num_pages: int = 2, 
    save_interval: int = 4, 
    optimizer_lr: float = 4e-4, 
    optimizer_beta1: float = 0.9, 
    optimizer_beta2: float = 0.95, 
    optimizer_weight_decay: float = 0.1,
    use_wandb: bool = False,
    use_batch_norm: bool = False
):
    """
    Main function to train the GPT-2 model with specified hyperparameters.

    Args:
        run_name (str): Wandb run name.
        project_name (str): Wandb project name.
        model_name (str): Name of the model to use.
        batch_size (int): Size of each batch.
        sequence_length (int): Length of each sequence.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to use for training (e.g., 'cuda:1').
        num_pages (int): Number of pages to load from the dataset.
        save_interval (int): Interval to save the model.
        optimizer_lr (float): Learning rate for the AdamW optimizer.
        optimizer_betas1 (float): Betas1 for the AdamW optimizer.
        optimizer_betas2 (float): Betas2 for the AdamW optimizer.
        optimizer_weight_decay (float): Weight decay for the AdamW optimizer.
        method: (str): baseline, seqcompress, batchcompress
        use_wandb (bool): Whether to use wandb for logging.
        use_batch_norm (bool): use batch normalization at grad level.
    """
    args = SimpleNamespace(
        run_name = run_name,
        project_name = project_name,
        model_name=model_name, 
        batch_size=batch_size, 
        sequence_length=sequence_length, 
        learning_rate=learning_rate, 
        device=device, 
        num_pages=num_pages, 
        save_interval=save_interval, 
        optimizer_beta1=optimizer_beta1,
        optimizer_beta2=optimizer_beta2,
        optimizer_weight_decay=optimizer_weight_decay,
        method=method,
        use_wandb=use_wandb,
        use_batch_norm = use_batch_norm,
    )
    # Print the args to screen in a nicely formatted manner
    if args.use_wandb and args.run_name == None:
        print("Error: You need to set a unique run name i.e. --run-name baseline-1")
        sys.exit()
    if args.method not in AVAILABLE_METHODS:
        print(f"Error: You need to set a method in {AVAILABLE_METHODS}")
        sys.exit()

    print("Training Configuration:")
    print(f"Run Name: {args.run_name}")
    print(f"Project Name: {args.project_name}")
    print(f"Method: {args.method}")
    print(f"Model Name: {args.model_name}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Number of Pages: {args.num_pages}")
    print(f"Save Interval: {args.save_interval}")
    print(f"Optimizer Beta1: {args.optimizer_beta1}")
    print(f"Optimizer Beta2: {args.optimizer_beta2}")
    print(f"Optimizer Weight Decay: {args.optimizer_weight_decay}")
    print(f"Use WandB: {args.use_wandb}")

    # Load the tokenizer
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(args.model_name, verbose=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Create a GPT2 model from scratch.
    configuration = GPT2Config(output_hidden_states=False, n_positions=sequence_length)
    model = GPT2LMHeadModel(config=configuration)
    # model.load_state_dict(torch.load('/home/setup/reduct/fineweb_model.pt'))

    # Move model to the appropriate device
    device = args.device
    model.to(device)

    # AdamW optimizer with specified parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,  # Peak learning rate
        betas=(args.optimizer_beta1, args.optimizer_beta2), # B1 and B2
        weight_decay=args.optimizer_weight_decay  # Weight decay
    )

    inputs_dict = {}
    grad_inputs_dict = {}
    
    def save_layer_inputs(layer):
        """
        Hook to save the inputs of a layer during the forward pass.
        """
        def hook(module, input, output):
            if isinstance(input, tuple):
                if len(input) > 0:
                    inputs_dict[layer] = input[0].detach().clone()
            else:
                if input is not None:
                    inputs_dict[layer] = input.detach().clone()
        return hook
    
    def save_layer_grads(layer):
        """
        Hook to save the gradients of a layer during the backward pass.
        """
        def hook(module, grad_input, grad_output):
            grad_inputs_dict[layer] = grad_output[0].detach().clone()
        return hook
    
    # Register hooks for all layers that require gradients
    for name, layer in model.named_modules():
        if any(p.requires_grad for p in layer.parameters()):
            layer.register_forward_hook(save_layer_inputs(layer))
            layer.register_full_backward_hook(save_layer_grads(layer))
            


    def train_step_with_avg_grad(batch):
        """
        Perform a single training step using averaged gradient computation.
        """
        optimizer.zero_grad()

        # Shift the input ids and create labels
        input_ids = torch.tensor(batch, dtype=torch.long).to(device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = tokenizer.pad_token_id

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        
        if args.method == 'baseline':  
            all_realized_bandwidth = sum(param.grad.element_size() * param.grad.nelement() for param in model.parameters() if param.grad is not None)
            base_line_bandwidth = sum(param.grad.element_size() * param.grad.nelement() for param in model.parameters() if param.grad is not None)
        else:  
            optimizer.zero_grad()
            all_realized_bandwidth = 0
            base_line_bandwidth = 0
            for layer in list(grad_inputs_dict.keys()):
                
                inputs = inputs_dict[layer].to(device)
                grad_outputs = grad_inputs_dict[layer].to(device)

                # Check if the inputs can require gradients
                if inputs.requires_grad is False:
                    try:
                        inputs.requires_grad_(True)
                    except Exception as e:
                        pass
                layer_output = layer(inputs)[0]
                if isinstance(layer_output, tuple):
                    layer_output = layer_output[0]
                else:
                    layer_output = layer_output

                prev_size = grad_outputs.nelement()
                
                if args.use_batch_norm:
                    scaling_factor =  1 / (args.batch_size * args.sequence_length)
                    avg_layer_output = layer_output.view(-1, layer_output.shape[-1]).sum(dim=0, keepdim=True)
                    avg_grad_outputs = grad_outputs.view(-1, grad_outputs.shape[-1]).sum(dim=0, keepdim=True)
                else:
                    scaling_factor = (args.batch_size * args.sequence_length)
                    avg_layer_output = layer_output.view(-1, layer_output.shape[-1]).mean(dim=0, keepdim=True)
                    avg_grad_outputs = grad_outputs.view(-1, grad_outputs.shape[-1]).mean(dim=0, keepdim=True)
                
                post_size = avg_grad_outputs.nelement()
                
                all_realized_bandwidth += avg_layer_output.element_size() * avg_layer_output.nelement() + avg_grad_outputs.element_size() * avg_grad_outputs.nelement()

                # Recompute gradient w.r.t the layer using averaged values
                grad_wrt_layer_recomputed = torch.autograd.grad(outputs=avg_layer_output, inputs=layer.parameters(), grad_outputs=avg_grad_outputs, retain_graph=False)
                for grad in grad_wrt_layer_recomputed:
                    base_line_bandwidth += grad.element_size() * grad.nelement()

                # Scale the gradient to approximate the sum of gradients
                for i, param in enumerate(layer.parameters()):
                    param.grad = grad_wrt_layer_recomputed[i] * scaling_factor
                    
        # Apply the gradients.
        optimizer.step()
        optimizer.zero_grad()

        return loss, all_realized_bandwidth, base_line_bandwidth

    # Initialize wandb if use_wandb is True
    if args.use_wandb:
        wandb.init(project=args.project_name, name = args.run_name, config=vars(args))
        
        
    # def logit_saving(batch):
    #      # Shift the input ids and create labels
    #     input_ids = torch.tensor(batch, dtype=torch.long).to(device)
    #     labels = input_ids.clone()
    #     labels[:, :-1] = input_ids[:, 1:]
    #     labels[:, -1] = tokenizer.pad_token_id

    #     # Forward pass
    #     outputs = model(input_ids=input_ids, labels=labels)
    #     loss = outputs.loss

    #     # Retain gradients w.r.t to the logits
    #     logits = outputs.logits
    #     logits.retain_grad()

    #     # Backward pass
    #     loss.backward()
        
    #     # get the gradients from the logits.
    #     logits_grads = logits.grad.clone()
        
    #     # Zero the grads.
    #     optimizer.zero_grad()

    #     # Free the graph and grads before this
    #     del logits.grad
    #     torch.cuda.empty_cache()

    #     # Do the full backward using the logits grads and the inputs
    #     # Perform the full backward pass using the logits gradients and the inputs
    #     final_layer_recreated = model(input_ids=input_ids, labels=labels).logits
    #     avg_grad_final_layer = logits_grads.mean(dim=0) * args.batch_size
    #     final_layer_recreated_compressed = final_layer_recreated.mean(dim=0)

    #     # Backward pass using the retained gradients
    #     final_layer_recreated_compressed.backward(gradient=avg_grad_final_layer)
        
    #     # Apply the step.
    #     optimizer.step()
        
    #     # compute the bandwidth wins.
    #     all_realized_bandwidth = avg_grad_final_layer.element_size() * avg_grad_final_layer.nelement() 
    #     base_line_bandwidth = sum(param.grad.element_size() * param.grad.nelement() for param in model.parameters() if param.grad is not None)

    #     # Zero grads.
    #     optimizer.zero_grad()
    #     return loss, all_realized_bandwidth, base_line_bandwidth


    model.train()
    while True:
        try:
            dataset = SubsetFineWebEdu2Loader(batch_size=batch_size, sequence_length=sequence_length, num_pages=args.num_pages, tokenizer=tokenizer)
            for idx, batch in enumerate(dataset):
                loss, realized_bandwidth, base_line_bandwidth = train_step_with_avg_grad(batch)
                # loss, realized_bandwidth, base_line_bandwidth = logit_saving( batch )
                print(f"{args.method} - {args.run_name} - Loss: {loss.item()}, reduction_x: {base_line_bandwidth/realized_bandwidth}, perplexity: {math.exp(loss.item())}")
                
                # Log metrics to wandb if use_wandb is True
                if args.use_wandb:
                    wandb.log({
                        "Loss": loss.item(),
                        "Reduction_x": base_line_bandwidth/realized_bandwidth,
                        "Perplexity": math.exp(loss.item())
                    })
                                            
        except Exception as e:
            import traceback
            print(f"An error occurred during training step: {e}. Continuing training...")
            traceback.print_exc()
                    
        except KeyboardInterrupt:
            print("Training interrupted. Finishing wandb run.")
            if args.use_wandb:
                wandb.finish()
            break

if __name__ == "__main__":
    typer.run(main)
