# Script introduces the many concepts of training the GPT model using a toy nn model
# Model simply predicts the next character

import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyper-parameters
block_size = 8
batch_size = 4
max_iters = 100
eval_interval = 20
learning_rate = 3e-4
eval_iters = 250
file_path = "D:\gitFolders\pytorch_hardway\data\wizard_of_oz.txt"
# file processing to get text data
with open(file=file_path,
          mode='r',
          encoding='utf-8') as f:
    text = f.read()
# getting the different characters used in the text-corpus
chars = sorted(set(text))
# getting the total vocab_size, that is the number of unique characters
vocab_size = len(chars)

string_int = {ch: i for i, ch in enumerate(chars)}
int_string = {i: ch for i, ch in enumerate(chars)}


# declaring encoding and decoding functions
def encode(s):
    return [string_int[ch] for ch in s]


def decode(l): 
    return [int_string[i] for i in l]


# making the data
data = torch.tensor(encode(text), dtype=torch.long)
# split the traint and test data
n = int(0.8 * len(data))
train = data[:n]
test = data[n:]


# making the batches for training and testing
def get_batch(split: str):
    data = train if split == 'train' else test
    # create tensor with random numbers equal to batch_size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    # make the input data from random starting points inside the 
    # corpus and extract till the block size, 
    # stack batch_size number of data into X
    x = torch.stack([data[i: i + block_size] for i in ix])
    # do the same thing as X, but add 1 to blocksize so the model can
    # learn
    y = torch.stack([data[i: i + block_size + 1] for i in ix])
    # print(x.shape, y.shape)
    x, y = x.to(device), y.to(device)
    return x, y


x_train, y_train = get_batch('train')

@torch.no_grad()
def estimate_loss():
    "Estimate the losses for each split"
    out = {}
    # enable eval mode
    model.eval()
    # enumerate over train and test data
    for split in ['train', 'val']:
        # declare losses as 0s
        losses = torch.zeros(eval_iters)
        # enumerate over the number of evaluation iterations
        for k in range(eval_iters):
            # get a new random batch
            X, Y = get_batch(split)
            # get the logits and losses from model
            logits, loss = model(X, Y)
            # assign the loss to respective index in losses array
            losses[k] = loss.item()
        # get the mean losses for train and test
        out[split] = losses.mean()
    model.train()
    return out


# design the model
class BigramLangModel(nn.Module):
    # initialize the embedding table of size of vocab_size
    def __init__(self, vocab_size):
        super(BigramLangModel, self).__init__()
        # table will contain vocab_size of elements, with each 
        # element containing vocab_size of embedding element.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)

        if targets is None:
            loss = None
        
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        # need to understand how the logits look 
        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # ask for the prediction from forward method
            logits, loss = self.forward(index)
            # take only the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)
            # append the sampled index to running index
            index = torch.cat((index, index_next), dim=0)
        return index
    
model = BigramLangModel(vocab_size)
m = model.to(device)

# run the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
gen_chars = decode(m.generate(context, max_new_tokens=250)[0].tolist())
print(gen_chars)
