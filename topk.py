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

# Argument parser for hyperparameters
def main(model_name: str = 'gpt2', 
    run_name: str = None,
    project_name: str = 'fineweb-tuning',
    method: str = 'baseline',
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
    moving: bool = True,
    compression: int = 100,
    alpha: float = 0.1
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
        moving = moving,
        compression = compression,
        alpha = alpha,
    )
    # Print the args to screen in a nicely formatted manner
    if args.use_wandb and args.run_name == None:
        print("Error: You need to set a unique run name i.e. --run-name baseline-1")
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

    # Dictionary to maintain a moving average of the true gradients
    previous_gradients = {}

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
             
        all_realized_bandwidth = 0
        base_line_bandwidth = 0
        for name, layer in model.named_modules():
            for i, param in enumerate(layer.parameters()):
                
                moving = args.moving
                compression = args.compression
                # if moving == True: 
                    
                # Update moving average of the true gradient
                alpha = args.alpha
                prev_grad = param.grad.clone()
                if param in previous_gradients:
                    grad_variance = ((prev_grad - previous_gradients[param]) ** 2).view(-1)
                    grad_variance_normalized = torch.softmax(grad_variance, dim=0)
                    grad_entropy = grad_variance_normalized * torch.log(grad_variance_normalized + 1e-12)
                    grad_entropy_normalized = grad_entropy / grad_entropy.sum()
                    grad_absolute_normalized = prev_grad.abs().view(-1) / prev_grad.sum() 

                    if args.method == 'entva': 
                        gradient_score = grad_entropy_normalized + grad_variance_normalized  + grad_absolute_normalized
                    
                    # elif args.method == 'entv':
                    #     gradient_score = grad_entropy_normalized + grad_variance_normalized
                    
                    elif args.method == 'enta':
                        gradient_score = grad_entropy_normalized + grad_absolute_normalized

                    elif args.method == 'av':
                        gradient_score = grad_absolute_normalized + grad_variance_normalized

                    # elif args.method == 'v':
                    #     gradient_score = grad_variance_normalized
                        
                    elif args.method == 'a':
                        gradient_score = grad_absolute_normalized
                    
                    # elif args.method == 'ent':
                    #     gradient_score = grad_entropy_normalized
                        
                    elif args.method == 'absa':
                        gradient_score = prev_grad.abs().view(-1) 

                    elif args.method == 'absav':
                        gradient_score = prev_grad.abs().view(-1) + grad_variance

                    elif args.method == 'absavent':
                        gradient_score = prev_grad.abs().view(-1) + grad_variance + grad_entropy

                    previous_gradients[param] = prev_grad
                else:
                    gradient_score = prev_grad
                    previous_gradients[param] = prev_grad
                    
                
                if param.grad.nelement() > compression * 4:
                    topk_values, topk_indices = torch.topk( gradient_score.view(-1), k=int(max(1, gradient_score.nelement() * (1/compression))) )
                    base_line_bandwidth += param.grad.element_size() * param.grad.nelement()
                    param.grad.zero_()
                    gathered_values = prev_grad.view(-1).gather(0, topk_indices)
                    param.grad.view(-1).scatter_(0, topk_indices, gathered_values)
                    all_realized_bandwidth += topk_indices.element_size() * topk_indices.nelement() + topk_values.element_size() * topk_values.nelement()
                else:
                    base_line_bandwidth += param.grad.element_size() * param.grad.nelement()
                    all_realized_bandwidth += param.grad.element_size() * param.grad.nelement()            
                    
        # Apply the gradients.
        optimizer.step()
        optimizer.zero_grad()

        return loss, all_realized_bandwidth, base_line_bandwidth

    # Initialize wandb if use_wandb is True
    if args.use_wandb:
        wandb.init(project=args.project_name, name = args.run_name, config=vars(args))

    model.train()
    while True:
        try:
            dataset = SubsetFineWebEdu2Loader(batch_size=batch_size, sequence_length=sequence_length, num_pages=args.num_pages, tokenizer=tokenizer)
            for idx, batch in enumerate(dataset):
                loss, realized_bandwidth, base_line_bandwidth = train_step_with_avg_grad(batch)
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
