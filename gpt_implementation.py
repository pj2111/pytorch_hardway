# Implementing the complete GPT model from scratch by 
# building the layers step by step. Much of the data 
# extraction & processing will remain same with Bigram. 

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import pickle
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/gpt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# hyper-parameters
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 4
n_layers = 4
dropout = 0.2

file_path = "D:\gitFolders\pytorch_hardway\data\wizard_of_oz.txt" if device == 'cpu' else "/home/kamal/gitfolders/pytorch_hardway/data/wizard_of_oz.txt"
# extract data
with open(file=file_path, mode='r', encoding='utf-8') as raw:
    text = raw.read()
    chars = sorted(list(set(text)))


vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [string_to_int[c] for c in s]


def decode(l):
    return ''.join([int_to_string[i] for i in l])


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


# Model visualisation will be updated in the drawio file, refer that
# building the model from the head
class Head(nn.Module):
    """One head for Self-Attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,
                                                           block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size (batch, time-step, channels)
        # output size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Putting multiple heads in parallel with self-attention."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        # ModuleList works in parallel
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3, h4, h4, h4, h4]
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple linear layer followed by non-linearity like ReLU"""
    def __init__(self, n_embed):
        super().__init__()
        # sequential will go one layer after another
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    """Bringing it all together"""
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets = None):
        B, T = index.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(index)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index


model = GPTLanguageModel(vocab_size)
m = model.to(device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# writer.add_graph(model=model, input_to_model=[vocab_size])
# writer.close()
for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(torch.cuda.memory_allocated() / 1024)
print(loss.item())


prompt = 'Hello! Can you see me?'
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=500)[0].tolist())
print(generated_chars)