"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import torch
import argparse
from torchvision import transforms
import data_setup, engine, model_builder, utils

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Train a PyTorch image classification model.")

    # Add arguments
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument("--hidden_units", type=int, default=10, help="Number of hidden units in the model.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--train_dir", type=str, default="data/pizza_steak_sushi/train", help="Directory containing training data.")
    parser.add_argument("--test_dir", type=str, default="data/pizza_steak_sushi/test", help="Directory containing testing data.")
    parser.add_argument("--model_save_dir", type=str, default="models", help="Directory to save the trained model.")
    parser.add_argument("--model_save_name", type=str, default="05_going_modular_script_mode_tinyvgg_model.pth", help="Filename to save the trained model.")

    # Parse arguments
    args = parser.parse_args()

    # Setup hyperparameters
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate

    # Setup directories
    train_dir = args.train_dir
    test_dir = args.test_dir
    model_save_dir = args.model_save_dir
    model_save_name = args.model_save_name

    # Setup target device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create transform
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create DataLoader with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=LEARNING_RATE)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir=model_save_dir,
                     model_name=model_save_name)

if __name__ == "__main__":
    main()
