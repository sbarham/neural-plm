#############
## Imports ##
#############

import nltk
import spacy

import tqdm

import random
import pickle
import math
import pathlib

import torch


#######################
## Important Globals ##
#######################

N = 1000

nlp = spacy.load('en_core_web_sm')

num_tok = '<NUM>'
pad_tok = '<PAD>'


##########################
## Initialize the Rando ##
##########################

random.seed(777)


#######################
## The Dataset Class ##
#######################

class Dataset:
    def __init__(self, strings, train_test_split=.8):
        """
        strings : a list of sentences, or lines, or paragraphs -- strings of characters longer than a word
        """
        self._sentences = strings

        split_idx = math.ceil(len(strings) * .8)

        
        print("Preprocessing the corpus ...")
        self._preprocessed = [self._preprocess(sent) for sent in tqdm.tqdm(self._sentences)]
        
        self._words = [word for sent in self._preprocessed for word in sent.split()]
        
        self._train_words = [word for sent in self._preprocessed[:split_idx] for word in sent.split()]
        self._test_words = [word for sent in self._preprocessed[split_idx:] for word in sent.split()]

        self._partitions = {
            'all': self._words,
            'train': self._train_words,
            'test': self._test_words
        }

        
        print("Extracting vocabulary ...")
        print("   [Phase 1 of 2]")
        self._i2w = list({word for word in tqdm.tqdm(self._words)})
        print("   [Phase 2 of 2]")
        self._w2i = {word: idx for idx, word in tqdm.tqdm(enumerate(self._i2w))}

        # record the size of the vocabulary
        self.vocab_size = len(self._i2w)

    # Utilities
        
    def _preprocess(self, sent):
        sent = nlp(sent)
        sent = [num_tok if tok.is_digit else tok.text.lower() for tok in sent if not tok.is_punct]
        
        return ' '.join(sent)

    # API
    
    def idx2example(self, idx):
        return self._preprocessed[idx]

    def i2w(self, idx):
        return self._i2w[idx]

    def w2i(self, word):
        return self._w2i[word]

    def sample2idx(self, sample):
        """
        sample : list of strings
        """
        return [self.w2i(word) for word in sample]

    def idx2sample(self, sample):
        """
        sample : list of indexes into the vocabulary
        """
        return [self.i2w(word) for word in sample]
    
    def get_sample(self, sample_length, partition='train'):
        """
        sample_length : int denoting the number of words in the sample
        partition : string indicating whether to draw the sample from `all`, `train`, or `test`

        Randomly sample a string of length `sample_length` from the indicated dataset partition.
        The string returned is a list of indexes into the vocabulary.
        """
        idx = random.randint(0, len(self._partitions[partition]) - sample_length)

        sample = self._partitions[partition][idx:(idx + sample_length)]
        sample = self.sample2idx(sample)

        return sample

    def get_batch(self, sample_length, batch_size):
        batch = [self.get_sample(sample_length + 1) for _ in range(batch_size)]
        xs, ys = list(zip(*[(sample[:-1], sample[-1]) for sample in batch]))

        return torch.LongTensor(xs), torch.LongTensor(ys)
    


# Example Usage: Representing the Brown Corpus
    
brown_sentences = [' '.join(sent) for sent in tqdm.tqdm(nltk.corpus.brown.sents())][:N]

dataset_path = pathlib.Path('dataset.pkl')

# Don't create the Brown dataset every time; try and load it from this directory
# if possible

if dataset_path.exists():
    with open(dataset_path, 'rb') as f:
        brown_dataset = pickle.load(f)
else:
    brown_dataset = Dataset(brown_sentences)
    with open(dataset_path, 'wb') as f:
        pickle.dump(brown_dataset, f)
