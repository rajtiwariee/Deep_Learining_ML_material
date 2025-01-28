
"""
Contains functions for training and testing a Pytorch model
"""

import torch

from tqdm.auto import tqdm
from typing import Dict,List, Tuple
from timeit import default_timer as timer
from torch import nn

#setup the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model:torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
  #turn the model in train mode
  model.train()

  #setup train loss and train accuracy
  train_loss, train_accuracy = 0,0

  #Loop through the batch and train
  for batch, (X,y) in enumerate(dataloader):
    #set the data to same device
    X = X.to(device)
    y = y.to(device)

    #do the forward pass
    y_logits = model(X)

    #calculate the loss
    loss = loss_fn(y_logits, y)
    train_loss += loss

    #Optimizer zero grad
    optimizer.zero_grad()

    #backpropagation (calculate the gradients)
    loss.backward()

    #Optimizer step step step (reduce the loss)
    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_logits , dim = 1), dim = 1)
    #calculate the accuracy
    train_accuracy += (y_pred_class == y).sum().item()/ len(y_logits)


  #calculate accuracy per batch
  train_loss /= len(dataloader)
  train_accuracy /= len(dataloader)

  return train_loss, train_accuracy

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn : nn.Module,
              device: torch.device = device) ->Tuple(float, float):
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
  #set the model in eval mode
  model.eval()

  #Setup test loss , test accuracy
  test_loss, test_accuracy = 0,0

  #pass the data through the model
  with torch.inference_mode():

    for X,y in dataloader:

      #set the data to same device
      X = X.to(device)
      y = y.to(device)

      #1.do the forward pass
      y_pred = model(X)

      #2. calculate the loss
      loss = loss_fn(y_pred, y)
      test_loss += loss

      #calculate the accuracy
      test_pred_class = torch.softmax(y_pred, dim = 1).argmax(dim = 1)
      test_accuracy += (test_pred_class == y).sum().item() / len(y_pred)

  #calculate the loss and accuracy per batch
  test_loss /= len(dataloader)
  test_accuracy /= len(dataloader)

  return test_loss, test_accuracy


def train(model:torch.nn.Module,
          train_dataloader : torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          EPOCHS: int,
          loss_fn: nn.Module = nn.CrossEntropyLoss(),
          device: torch.device = device):

  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
  """

  #create a result dictionary
  results = {'train_loss': [],
             'train_acc': [],
             'test_loss':[],
             'test_acc':[]}

  #put the model on the same device
  model.to(device)
  #Loop through training and test loop
  for epoch in tqdm(range(EPOCHS)):

      #train step
      train_loss, train_accuracy = train_step(model = model,
                                              dataloader = train_dataloader,
                                              loss_fn = loss_fn,
                                              optimizer = optimizer,
                                              device = device)

      #test step
      test_loss, test_accuracy = test_step(model = model,
                                           dataloader = test_dataloader,
                                           loss_fn = loss_fn,
                                           device = device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_accuracy:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_accuracy:.4f}"
      )

      #append the results
      results['train_loss'].append(train_loss)
      results['test_loss'].append(test_loss)
      results['train_acc'].append(train_accuracy)
      results['test_acc'].append(test_accuracy)

  return results
