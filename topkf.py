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
def main(model_name: str = 'llama', 
    run_name: str = None,
    project_name: str = 'fineweb-tuning',
    method: str = 'topk',
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
    alpha: float = 0.1,
    scoring: str = 'hessian',
    use_residual: bool = True,
):
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
        scoring = scoring,
        use_residual = use_residual,
    )
    # Print the args to screen in a nicely formatted manner
    if args.use_wandb and args.run_name == None:
        print("Error: You need to set a unique run name i.e. --run-name baseline-1")
        sys.exit()
    from pprint import pprint
    pprint(vars(args))

    # Load the tokenizer
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('gpt2', verbose=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Create a GPT2 model from scratch.
    if model_name == 'gpt2':
        configuration = GPT2Config(output_hidden_states=False, n_positions=sequence_length)
        model = GPT2LMHeadModel(config=configuration)
    elif model_name == 'llama':
        from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
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
    residuals = {}

    def train_step_with_topk(batch, previous_gradients, residuals ):
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
        
        args.scoring = 'hessian'
        if method == 'topk':
            optimizer.zero_grad() # Santity check.
            chunk_idx = 0
            max_chunk_size = 200_000_000
            all_grads = []
            current_chunk_size = 0
            all_realized_bandwidth = 0
            base_line_bandwidth = 0
            residuals[chunk_idx] = None
            for name, layer in model.named_modules():

                for i, param in enumerate(layer.parameters()):
                    
                    grad = param.grad.clone()
                    if args.scoring == 'hessian':
                        # Hessian diagonal approximation: H_ii ≈ (grad**2)
                        hessian_diag_approx = grad ** 2
                        
                        # Compute the salience score for each element in the gradient using the second-order approximation
                        grad_score = (0.5 * hessian_diag_approx * grad ** 2).view(-1)
                        
                    elif args.scoring == 'variance':
                    
                        if param not in previous_gradients:
                            previous_gradients[param] = grad
                        grad_variance = ((grad - previous_gradients[param]) ** 2).view(-1)
                        grad_score = grad.abs().view(-1) + grad_variance.view(-1)
                    
                    
                    all_grads.append((grad_score, param))
                    previous_gradients[param] = grad
                    current_chunk_size += grad.numel()
                    
                if current_chunk_size > max_chunk_size:
                    # Flatten all the gradients into a single [-1] vector.
                    all_scores_flat = torch.cat([g.view(-1) for g, _ in all_grads]) 

                    # Check if we have a residual and if use_residual flag is set. If we do add them to the all scores.
                    if args.use_residual and residuals[chunk_idx] != None:
                        all_scores_flat += residuals
                        
                    # Set the residual to the previous all scores (keep track of scores through time.)
                    residuals[chunk_idx] = all_scores_flat.clone()

                    # Flatten out the gradients themselves for topk indexing.
                    all_grads_flat = torch.cat([p.grad.view(-1) for _, p in all_grads])

                    # Compute the topk values and then compute the implied bandwidth.
                    topk_values, topk_indices = torch.topk(all_scores_flat, k=int(all_grads_flat.shape[-1]/args.compression))
                    all_realized_bandwidth += topk_values.nelement()
                    base_line_bandwidth += all_grads_flat.nelement()

                    # Zero out the residuals for the topk gradients if use_residual flag is set.
                    # This means that gradients can accumulate their gradients over time in the event that
                    # the topk removes them. If the gradient is used, their residual goes to zero.
                    # Otherwise, the grad score is additive every step.
                    if args.use_residual:
                        residuals[chunk_idx][topk_indices] = 0 

                    # Actually compute the topk gradients mapping them from the flat all grads.
                    topk_grads = torch.zeros_like(all_grads_flat)
                    topk_grads[topk_indices] = all_grads_flat[topk_indices]  # Map the topk indices back to the original parameters

                    # Reshape the topk gradients back to the original parameter shapes
                    start_idx = 0
                    for grad_score, param in all_grads:
                        numel = param.numel()
                        param.grad = topk_grads[start_idx:start_idx + numel].view(param.shape)
                        start_idx += numel
                    
                    # Reset chunk.
                    all_grads = []
                    current_chunk_size = 0
                    chunk_idx += 1
                    residuals[chunk_idx] = None
                    
        elif method == 'baseline':
            all_realized_bandwidth = 1
            base_line_bandwidth = 1                          
                    
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
                loss, realized_bandwidth, base_line_bandwidth = train_step_with_topk( batch, previous_gradients, residuals )
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
