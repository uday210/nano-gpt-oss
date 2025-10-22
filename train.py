"""
An updated training script for the Nano‑GPT‑OSS repository.

This script combines the data loading, model instantiation, training and
text generation steps into a single entry point.  It uses the existing
`training.data_loader` to build the TinyStories training and validation
datasets, the `training.trainer.trainer` function to run a simple
training loop, and the `inference.generate_text` function to produce
sample text from the trained model.  The generated text is printed to
the console after training completes.

To run this script, make sure all dependencies from the project's
`requirements.txt` are installed.  Because the full model and dataset
are large, you may wish to adjust the model configuration (e.g.
reduce `num_hidden_layers` or `hidden_size`) or use a smaller
dataset to avoid running out of memory.
"""

import torch

from training.data_loader import train_loader, val_loader  # data loaders
from architecture.gptoss import Transformer, ModelConfig  # model and config
from training.trainer import trainer  # training function
from inference import generate_text  # text generation helper


def main() -> None:
    """Entry point: instantiate model, train, and generate text."""
    # Select device based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model hyper‑parameters; adjust as necessary for your hardware
    model_cfg = ModelConfig(
        num_attention_heads=8,
        num_key_value_heads=4,
        num_experts=4,
        experts_per_token=1,
        num_hidden_layers=12,
        hidden_size=1024,
        intermediate_size=1024,
    )

    # Instantiate Transformer
    model = Transformer(model_cfg, device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model size: {total_params:.2f}M parameters")

    # Optionally load a saved checkpoint if one exists
    checkpoint_path = "model/gptoss.pt"
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print("No existing checkpoint found; training from scratch.")

    # Train the model using the provided trainer
    print("Starting training…")
    train_losses, val_losses, tokens_seen = trainer(model, train_loader, val_loader, device)
    print("Training finished.")

    # Save the trained model
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved trained model to {checkpoint_path}")

    # Generate sample text
    prompt = "Once upon a day"
    print(f"\nGenerating text for prompt: {prompt!r}\n")
    generated_text = generate_text(model, prompt, max_tokens=100)
    print(generated_text)


if __name__ == "__main__":
    main()
