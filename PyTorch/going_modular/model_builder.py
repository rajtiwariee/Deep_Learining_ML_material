
"""
Contains PyTorch model code to instantiate a TinyVGG model

"""

import torch
from torch import nn

class TinyVGG(nn.Module):
  """ Creates a TinyVGG architecture

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch

  Args:
    input_shape: An integer indicating number of input channels (color channels)
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.

  """
  def __init__(self, input_shape: int, hidden_units:int, output_shape:int):
    super().__init__()

    #layer stack 1
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels = input_shape,out_channels = hidden_units,
                  kernel_size = (3,3),
                  stride = 1,
                  padding = 0),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                  kernel_size = (3,3),
                  stride = 1,
                  padding = 0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2,2),
                     stride = 2,
                  )
    )

    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                  kernel_size = (3,3),
                  stride = 1,
                  padding = 0),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                  kernel_size = (3,3),
                  stride = 1,
                  padding = 0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2,2),
                     stride = 2,
                    )
    )

    self.classifier_layer = nn.Sequential(
        nn.Flatten(),
        # Where did this in_features shape come from?
        # It's because each layer of our network compresses and changes the shape of our inputs data.
        nn.Linear(in_features = hidden_units*13*13,
                  out_features = output_shape),
    )

  #overwrite pytorch forward layer
  def forward(self, x: torch.Tensor) -> torch.Tensor:

    return self.classifier_layer(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
