#!/usr/bin/env python

import math

import corpus
import model

import torch
import torch.optim as optim
import torch.nn as nn

# Globals
loss = nn.NLLLoss()

# Training Parameters
print_every = 100
num_iterations = int(1e5)

# Hyperparameters
sample_length = 100
batch_size = 16

# Get the dataset
dataset = corpus.brown_dataset

# Create the model
print("Initializing model ...")
lm = model.ProbabilisticNeuralLM(dataset.vocab_size, sample_length=sample_length)
optimizer = optim.Adam(lm.parameters(), amsgrad=True)

losses = []

print("Beginning training ...")
for i in range(num_iterations):
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Get the next batch
    xs, ys = dataset.get_batch(sample_length, batch_size)

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

        print("({}) : {}".format(i, perp))
