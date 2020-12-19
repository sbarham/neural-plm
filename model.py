#############
## Imports ##
#############

import random

import torch
import torch.nn as nn

import corpus


#######################
## Important Globals ##
#######################

sample_length = 100
emb_dim = 100
hidden_dim = 500


################################
## Dataset-Dependent Defaults ##
################################

dataset = corpus.brown_dataset
vocab_size = dataset.vocab_size


##########################
## Initialize the Rando ##
##########################

random.seed(777)


#####################
## The Model Class ##
#####################

class ProbabilisticNeuralLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_dim=emb_dim,
            sample_length=sample_length,
            hidden_dim=hidden_dim
    ):
        super(ProbabilisticNeuralLM, self).__init__()
        
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.hidden = nn.Linear(sample_length * emb_dim, hidden_dim)
        self.hidden_activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.output_activation = nn.LogSoftmax(dim=-1)

    def forward(self, xs):
        batch_size = xs.size()[0]

        # embed and merge
        xs = self.emb(xs)
        xs = torch.reshape(xs, (batch_size, -1))

        # hidden layer
        hidden = self.hidden(xs)
        hidden = self.hidden_activation(hidden)

        # output log probabilities
        output_logits = self.output(hidden)
        output_log_probs = self.output_activation(output_logits)
        
        return output_log_probs

    def predict(self, xs):
        return torch.argmax(self.forward(xs))
