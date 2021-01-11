##
# TODO
#    * introduce padding (especially in iter_batches())
#
##



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

from collections import Counter, defaultdict

import torch


#######################
## Important Globals ##
#######################

N = 1000

nlp = spacy.load('en_core_web_sm')

unk_tok = '<UNK>'
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
    def __init__(self, strings, train_test_split=.8, min_freq=1):
        """
        strings : a list of sentences, or lines, or paragraphs -- strings of characters longer than a word
        """
        self._sentences = strings

        split_idx = math.ceil(len(strings) * .8)

        
        print("Preprocessing the corpus ...")
        self._preprocessed = [self._preprocess(sent) for sent in tqdm.tqdm(self._sentences)]
        
        print("Creating dataset partitions ...")
        self._words = [word for sent in self._preprocessed for word in sent.split()]
        
        self._train_words = [word for sent in self._preprocessed[:split_idx] for word in sent.split()]
        self._test_words = [word for sent in self._preprocessed[split_idx:] for word in sent.split()]

        self._partitions = {
            'all': self._words,
            'train': self._train_words,
            'test': self._test_words
        }

        if min_freq > 1:
            print("Collecting word type counts ...")
            self._word_freq = Counter(self._words)
            
            print("Filtering out low frequency words ...")
            self._i2w = [unk_tok] + [word for word, freq in self._word_freq.items() if freq >= min_freq]
        else:
            print("Extracting vocabulary ...")
            self._i2w = [unk_tok] + list({word for word in tqdm.tqdm(self._words)})
            
        print("Constructing word-to-index mapping ...")
        self._w2i = defaultdict(int)
        self._w2i.update({word: idx for idx, word in tqdm.tqdm(enumerate(self._i2w), total=len(self._i2w))})

        # record the size of the vocabulary
        self.vocab_size = len(self._i2w)

    # Utilities
        
    def _preprocess(self, sent):
        sent = nlp(sent)
        sent = [num_tok if (tok.is_digit or tok.like_num or tok.pos_ == 'NUM') else tok.text.lower() \
                for tok in sent \
                if not (tok.is_punct or tok.text == '`')]
        
        return ' '.join(sent)


    #####################
    ## Vocab Utilities ##
    #####################
    
    def i2w(self, idx):
        return self._i2w[idx]

    def w2i(self, word):
        return self._w2i[word]
    
    ######################
    ## Sample Utilities ##
    ######################

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

    ########################
    ## Batching Utilities ##
    ########################
    
    def samples2input_label(self, samples):
        return list(zip(*[(sample[:-1], sample[-1]) for sample in samples]))

    def batch2tensor(self, batch):
        return torch.LongTensor(batch[0]), torch.LongTensor(batch[1])
    
    #########################################
    ## Iterate Through a Dataset Partition ##
    ##    in order -- one by one           ##  
    #########################################
    
    def get_num_batches(self, sample_length, batch_size, partition='train'):
        words_in_batch = sample_length * batch_size
        data = self._partitions[partition]
        
        return math.floor(len(data) / words_in_batch)
    
    def get_next_batch(self, idx, sample_length, batch_size, partition='train'):
        partition = self._partitions[partition]

        start_idx = idx * sample_length * batch_size

        samples = [partition[start_idx + (batch * sample_length) : start_idx + ((batch + 1) * sample_length)] \
                for batch in range(batch_size)]

        samples = [self.sample2idx(sample) for sample in samples]
        
        return self.batch2tensor(self.samples2input_label(samples))

    
    def iter_batch(self, sample_length, batch_size, partition='train'):
        num_batches = self.get_num_batches(sample_length + 1, batch_size, partition)
        
        return (self.get_next_batch(idx, sample_length + 1, batch_size, partition) for idx in range(num_batches))

    #############################################
    ## Sample Batches From a Dataset Partition ##
    ##    each sample in the batch drawn at    ##
    ##    at random                            ##
    #############################################
    
    def get_batch(self, sample_length, batch_size, partition='train'):
        samples = [self.get_sample(sample_length + 1, partition) for _ in range(batch_size)]
        
        return self.batch2tensor(self.samples2input_label(samples))

    


# Example Usage: Representing the Brown Corpus
    
brown_sentences = [' '.join(sent) for sent in tqdm.tqdm(nltk.corpus.brown.sents())]

dataset_path = pathlib.Path('dataset.pkl')

# Don't create the Brown dataset every time; try and load it from this directory
# if possible

if dataset_path.exists():
    with open(dataset_path, 'rb') as f:
        brown_dataset = pickle.load(f)
else:
    brown_dataset = Dataset(brown_sentences, min_freq=3)
    with open(dataset_path, 'wb') as f:
        pickle.dump(brown_dataset, f)
