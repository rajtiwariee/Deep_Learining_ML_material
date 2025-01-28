"""
Contains various utility functions for pytorch model training and saving

"""

import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """

  #create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents = True, exist_ok = True)

  #create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  #save the model state_dict()

  print(f'[INFO] Saving model to: {model_save_path}')
  torch.save(obj = model.save_dict(),
             f = model_save_path)

def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Load the model weights from a specified path.

    Args:
        model (torch.nn.Module): The model architecture into which the weights will be loaded.
        model_path (str): The path to the file containing the saved model weights.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    # Load the state dictionary from the file
    state_dict = torch.load(model_path)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    return model
