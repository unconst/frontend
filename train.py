import os
import math
import torch
import wandb
import typer
import torch.optim as optim
from types import SimpleNamespace
from transformers import AutoTokenizer
from dataset import SubsetFineWebEdu2Loader
from transformers import GPT2Config, GPT2LMHeadModel, AdamW

# Argument parser for hyperparameters
def main(model_name: str = 'gpt2', 
    project_name: str = 'fineweb-tuning',
    batch_size: int = 2, 
    sequence_length: int = 1024, 
    learning_rate: float = 5e-7, 
    device: str = 'cuda:1', 
    num_pages: int = 2, 
    save_interval: int = 4, 
    look_ahead: bool = True,
    optimizer_lr: float = 4e-4, 
    optimizer_beta1: float = 0.9, 
    optimizer_beta2: float = 0.95, 
    optimizer_weight_decay: float = 0.1,
    baseline: bool = False,
    use_wandb: bool = False
):
    """
    Main function to train the GPT-2 model with specified hyperparameters.

    Args:
        project_name (str): Wandb project name.
        model_name (str): Name of the model to use.
        batch_size (int): Size of each batch.
        sequence_length (int): Length of each sequence.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to use for training (e.g., 'cuda:1').
        num_pages (int): Number of pages to load from the dataset.
        save_interval (int): Interval to save the model.
        look_ahead (bool): Whether to use look-ahead mechanism for gradient computation.
        optimizer_lr (float): Learning rate for the AdamW optimizer.
        optimizer_betas1 (float): Betas1 for the AdamW optimizer.
        optimizer_betas2 (float): Betas2 for the AdamW optimizer.
        optimizer_weight_decay (float): Weight decay for the AdamW optimizer.
        baseline: (bool): gradients are applied normally.
        use_wandb (bool): Whether to use wandb for logging.
    """
    args = SimpleNamespace(
        project_name = project_name,
        model_name=model_name, 
        batch_size=batch_size, 
        sequence_length=sequence_length, 
        learning_rate=learning_rate, 
        device=device, 
        num_pages=num_pages, 
        save_interval=save_interval, 
        look_ahead=look_ahead,
        optimizer_beta1=optimizer_beta1,
        optimizer_beta2=optimizer_beta2,
        optimizer_weight_decay=optimizer_weight_decay,
        baseline=baseline,
        use_wandb=use_wandb
    )
    # Print the args to screen in a nicely formatted manner
    print("Training Configuration:")
    print(f"Project Name: {args.project_name}")
    print(f"Model Name: {args.model_name}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Number of Pages: {args.num_pages}")
    print(f"Save Interval: {args.save_interval}")
    print(f"Look Ahead: {args.look_ahead}")
    print(f"Optimizer Beta1: {args.optimizer_beta1}")
    print(f"Optimizer Beta2: {args.optimizer_beta2}")
    print(f"Optimizer Weight Decay: {args.optimizer_weight_decay}")
    print(f"Baseline: {args.baseline}")
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
        
        if args.baseline:
            # Use normal gradients train normally.
            all_realized_bandwidth = sum(param.grad.element_size() * param.grad.nelement() for param in model.parameters() if param.grad is not None)
            base_line_bandwidth = sum(param.grad.element_size() * param.grad.nelement() for param in model.parameters() if param.grad is not None)

        else:
            # Compute and apply averaged gradients
            all_realized_bandwidth = 0
            base_line_bandwidth = 0
            optimizer.zero_grad()
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

                if args.look_ahead:
                    avg_layer_output = layer_output.view(-1, layer_output.shape[-1]).mean(dim=0, keepdim=True)
                    avg_grad_outputs = grad_outputs.view(-1, grad_outputs.shape[-1]).mean(dim=0, keepdim=True)
                else:
                    if len(layer_output.shape) == 3:
                        avg_layer_output = layer_output.mean(dim=0, keepdim=True).squeeze(0)
                    else:
                        avg_layer_output = layer_output
                    avg_grad_outputs = grad_outputs.mean(dim=0, keepdim=True).squeeze(0)
            
                all_realized_bandwidth += avg_layer_output.element_size() * avg_layer_output.nelement() + avg_grad_outputs.element_size() * avg_grad_outputs.nelement()

                # Recompute gradient w.r.t the layer using averaged values
                grad_wrt_layer_recomputed = torch.autograd.grad(outputs=avg_layer_output, inputs=layer.parameters(), grad_outputs=avg_grad_outputs, retain_graph=False)
                for grad in grad_wrt_layer_recomputed:
                    base_line_bandwidth += grad.element_size() * grad.nelement()

                # Scale the gradient to approximate the sum of gradients
                for i, param in enumerate(layer.parameters()):
                    param.grad = grad_wrt_layer_recomputed[i] * batch_size

        # Apply the gradients.
        optimizer.step()
        optimizer.zero_grad()

        return loss, all_realized_bandwidth, base_line_bandwidth

    # Initialize wandb if use_wandb is True
    if args.use_wandb:
        wandb.init(project=args.project_name, config=vars(args))
        
    model.train()
    while True:
        try:
            dataset = SubsetFineWebEdu2Loader(batch_size=batch_size, sequence_length=sequence_length, num_pages=args.num_pages, tokenizer=tokenizer)
            for idx, batch in enumerate(dataset):
                loss, realized_bandwidth, base_line_bandwidth = train_step_with_avg_grad(batch)
                print(f"Loss: {loss.item()}, reduction_x: {base_line_bandwidth/realized_bandwidth}, perplexity: {math.exp(loss.item())}")
                
                # Log metrics to wandb if use_wandb is True
                if args.use_wandb:
                    wandb.log({
                        "Loss": loss.item(),
                        "Reduction_x": base_line_bandwidth/realized_bandwidth,
                        "Perplexity": math.exp(loss.item())
                    })
                
                # Save the model to the local directory every save_interval steps
                if idx % args.save_interval == 0:
                    model_save_path = f"./fineweb_model.pt"
                    torch.save(model.state_dict(), model_save_path)
                    print('saved new model.')
                    if args.use_wandb:
                        wandb.save(model_save_path)
                            
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
