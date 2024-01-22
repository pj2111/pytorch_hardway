# Script introduces the many concepts of training the GPT model using a toy nn model
# Model simply predicts the next character
import logging
# logging.basicConfig(level=logging.INFO, format='%(message)s')

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"  # doesn't help with the OOM error

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# hyper-parameters
block_size = 8
batch_size = 2
max_iters = 100
eval_interval = 20
learning_rate = 3e-4
eval_iters = 250
file_path = "D:\gitFolders\pytorch_hardway\data\wizard_of_oz.txt" if device == 'cpu' else "/home/kamal/gitfolders/pytorch_hardway/data/wizard_of_oz.txt" 
# file processing to get text data
with open(file=file_path,
          mode='r',
          encoding='utf-8') as f:
    text = f.read()
# getting the different characters used in the text-corpus
chars = sorted(set(text))
# print(chars)
# getting the total vocab_size, that is the number of unique characters
vocab_size = len(chars)
# print('vocab size', vocab_size / 3)

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
print(f'Data shape is: {data.shape}')
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
    y = torch.stack([data[i+1: i + block_size + 1] for i in ix])
    # print(x.shape, y.shape)
    x, y = x.to(device), y.to(device)
    return x, y


x_train, y_train = get_batch('train')
# print(x_train.shape, y_train.shape)
# print(x_train.tolist(), y_train.tolist())
# print(torch.cuda.memory_allocated()/1024)

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
            # print(X)
            # print(Y)
            # get the logits and losses from model
            logits, loss = model.forward(X, Y)
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
        # print('logits', logits)
        if targets is None:
            loss = None
        
        else:
            B, T, C = logits.shape
            # print(logits.shape)  # torch.Size([2, 8, 81])
            # print(targets.shape)  # torch.Size([2, 9])
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        # need to understand how the logits look 
        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in current context
        # print('memory before iteration', torch.cuda.memory_allocated()/1024)
        for ind in range(max_new_tokens):
            # ask for the prediction from forward method
            logits, loss = self.forward(index)
            # take only the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities, observe the dim argument
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)
            # print(index.shape)
            # print(index_next.shape)
            # append the sampled index to running index
            index = torch.cat((index, index_next), dim=1)
            # print(f'memory after {ind} loop', torch.cuda.memory_allocated()/1024)
        return index
    
model = BigramLangModel(vocab_size)
# print('memory after model load', torch.cuda.memory_allocated()/1024)
m = model.to(device)

# declare the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"""step: {iter},
              train loss: {losses['train']:.3f},
              val loss: {losses['val']:.3f}""")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# run the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
# print(vocab_size) 
# first look at the output for one token index
output_tensor = m.generate(context, max_new_tokens=10)[0]
list_output = output_tensor.tolist()
# print(len(list_output))
gen_chars = decode(list_output)
print(gen_chars)
