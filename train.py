#!/usr/bin/env python

import math
from pathlib import Path

import tqdm

import corpus
import model

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

# Globals
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss = nn.NLLLoss()

# Training Parameters
print_every = 500
test_every = 2000
adjust_every = 500
num_iterations = int(1e7)
epsilon_0 = 1e-3
r = 1e-8

# Hyperparameters
sample_length = 5 # 100
batch_size = 512

# Model Parameters
emb_dim = 60
hidden_dim = 50

# Tensorboard-related
experiment_name = Path('act-{}_emb-{}_hidden-{}_sample-{}_batch-{}_optim-{}_lr-{}_adjust-every-{}'.format(
    'relu',
    emb_dim,
    hidden_dim,
    sample_length,
    batch_size,
    'sgd',
    epsilon_0,
    adjust_every
))
experiment_path = Path('run') / experiment_name
writer = SummaryWriter(experiment_path)

# Get the dataset
dataset = corpus.brown_dataset
num_test_batches = dataset.get_num_batches(
    sample_length,
    batch_size,
    partition='test'
)

# Create the model
print("Initializing model ...")
lm = model.ProbabilisticNeuralLM(
    dataset.vocab_size,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    sample_length=sample_length
)
lm.to(device)
optimizer = optim.SGD(lm.parameters(), lr=epsilon_0) # optim.Adam(lm.parameters(), amsgrad=True)

losses = []

print("Beginning training ...")
for i in range(num_iterations):    
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Get the next batch
    xs, ys = dataset.get_batch(sample_length, batch_size)
    xs, ys = xs.to(device), ys.to(device)

    # Get the averaged batch loss
    output = loss(lm(xs), ys)

    # Run the backward pass (calculate gradients)
    output.backward()

    # Update the model
    optimizer.step()

    # Save and print statistics
    losses.append(output.item())

    if i % print_every == (print_every - 1):
        # Get average loss
        avg_loss = sum(losses) / len(losses)
        losses = []

        # Compute perplexity
        perp = math.exp(avg_loss)

        # Print and log the train loss
        print("({}) : {}".format((i + 1), perp))
        writer.add_scalar('Training Loss', perp, i)

    if i % test_every == (test_every - 1):
        print("\nEvaluating model on test set ...")
        
        lm.eval()

        with torch.no_grad():
            batches = dataset.iter_batch(
                sample_length,
                batch_size,
                partition='test'
            )

            test_losses = []            
            for xs, ys in tqdm.tqdm(batches, total=num_test_batches):
                xs, ys = xs.to(device), ys.to(device)
                output = loss(lm(xs), ys)
                test_losses.append(output.item())


        avg_test_loss = sum(test_losses) / len(test_losses)
        test_losses = []

        # Compute perplexity
        test_perp = math.exp(avg_test_loss)

        # Print and log the test loss
        print("Average Test Loss: {}\n".format(test_perp))
        writer.add_scalar('Test Loss', test_perp, i)
                
        lm.train()

    # Update the learning rate every iteration, just as in the paper
    if i % adjust_every == (adjust_every - 1):
        for g in optimizer.param_groups:
            g['lr'] = epsilon_0 / (1 + r * (i / adjust_every))
