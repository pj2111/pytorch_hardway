# Pytorch team had provided the
# below code and function in seperate 
# modules. Then used a jupyter-nb to 
# execute the functions.
import random
import math
import time
from io import open
import glob
import os
import unicodedata
import string
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

file_path = 'D:\\gitFolders\\pytorch_hardway\\data\\names_data\\names\\*.txt'
# print(findFiles(file_path))

all_letters = string.ascii_letters + ".,;'"
n_letter = len(all_letters)
category_lines = {}
all_categs = []
n_categs = len(all_categs)


def findFiles(path):
    return glob.glob(path)


# To turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
# print(unicodeToAscii('Ślusàrski'))  # Slusarski


# Read a file & split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(c) for c in lines]


# print(all_categs, n_categs)
# print(category_lines['Greek'][:3])

# Starting to convert the words into tensors
# To represent a single letter, we use a “one-hot vector” of size <1 x n_letters>. A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.


def letterToIndex(letter):
    """Returns the index of the letter inside 
    all_letters array"""
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letter)
    # in the first element, locate the 
    # letter's index and make it one
    # leave the rest as 0
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letter)
    # each line will contain characters, and each char 
    # has to be tensor with 1r X n_letter c. 
    for li, letter in enumerate(line):
        # Each char is one-hot encoded &
        # entire char is 1R X n_letter C tensor 
        # Char tensor is batched inside tensor
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# print(letterToIndex('T'))  # 34
# print(letterToTensor('T'))
""" LetterToTensor Output
[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0.]]
"""
# print(lineToTensor('Theodore'))
# print(n_letter)
# print(lineToTensor('Theodore').size())  # Size([8, 1, 56])


# This RNN module (mostly copied from the PyTorch for Torch users tutorial)
# is just 2 linear layers which operate on an input and hidden state, with a
# LogSoftmax layer after the output
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(in_features=input_size + hidden_size,
                             out_features=hidden_size)
        # hidden_size is for the tensor that needs to be predicted 
        # based on the input_size tensor that is provided by the 
        # current letter 
        self.h2o = nn.Linear(in_features=hidden_size,
                             out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # initing the hidden as zeros to begin with
        return Variable(torch.zeros(1, self.hidden_size))


def categFromOutput(output):
    """Returns the category and its index"""
    top_n, top_i = output.topk(1)  # Values=tensor([[1.8665]]), indices=tensor([[5]])
    category_i = top_i[0].item()  # 1.8665186378
    return all_categs[category_i], category_i
# print(categFromOutput(output))   # ('Polish', 12)


# accessing the training samples with ease
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    categ = randomChoice(all_categs)  # 0 to 18
    name = randomChoice(category_lines[categ])  # one of names from above categ 
    categ_tensor = torch.tensor([all_categs.index(categ)],
                                dtype=torch.long)
    name_tensor = lineToTensor(name)
    return categ, name, categ_tensor, name_tensor


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()  # get a hidden tensor

    rnn.zero_grad()  # zero the gradients
    # enumerate the tensors on the line_tensor
    for i in range(line_tensor.size()[0]):
        # predict the category & hidden tensor
        output, hidden = rnn(line_tensor[i],
                             hidden)
    # Compare the prediction with the target category
    loss = criterion(output, category_tensor)
    # Backpropagate
    loss.backward()
    # Update the paramters of the model
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() # return pred and loss


def timeSince(since):
    now = time.time()  # current time
    s = now - since  # difference
    m = math.floor(s / 60)  # calculate mins
    s -= m * 60  # remaining seconds
    return '%dm %ds' % (m, s)  # return string


# Starting Eval
def evaluate(name_tensor):
    hidden = rnn.initHidden()
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)
    return output
    # there is no back propagation


def predict(input_line, n_pred=3):
    print(f"> {input_line}")
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        topv, topi = output.topk(n_pred, 1, True)
        predictions = []

        for i in range(n_pred):
            value = topv[0][i].item()
            categ_index = topi[0][i].item()
            print(f"{value:.2f} {all_categs[categ_index]}")
            predictions.append(value, all_categs[categ_index])


if __name__ == '__main__':

    for filename in findFiles(file_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categs.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    # initing the RNN
    n_hidden = 128
    rnn = RNN(n_letter, n_hidden, n_categs)

    input = lineToTensor('Albert')  # Size([5, 1, 56])
    hidden = torch.zeros(1, n_hidden)

    # get the next letter and the next hidden
    output, next_hidden = rnn(input[0], hidden)
    # print(output)
    # print(next_hidden)
    # How is the next_hidden helping to find the categories?

    for i in range(10):
        category, name, category_tensor, name_tensor = randomTrainingExample()
        print('Category= ', category, '/ line: ', name)

    # setting up the criterion to measure the loss
    criterion = nn.NLLLoss()

    """ Training Loop:
    - Create input(name) and target(category) tensors
    - Create a zeroed initial hidden state
    - Read each letter in and
        + Keep hidden state for next letter
    - Compare final output to target
    - Back-propagate
    - Return the output and loss
    """

    # set a appropriate Learning Rate. Too High will explode and too low will 
    # not allow the model to learn
    learning_rate = 0.005

    n_iters = 1000
    print_every = 500
    plot_every = 1000

    # track loss
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, name, category_tensor, name_tensor = randomTrainingExample()
        output, loss = train(category_tensor, name_tensor)
        current_loss += loss

        # print iter number, loss, name n guess
        if iter % print_every == 0:
            guess, guess_i = categFromOutput(output)
            correct = '✔' if guess == category else '❌ (%s)' % category
            print(f"""{iter}, {iter/n_iters * 100},
                {timeSince(start)}, {loss}, {name},
                {guess}, {correct}""")

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every) 
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)
# We will create a confusion matrix, indicating for every actual language (rows) which language the network guesses (columns)
    confusion = torch.zeros(n_categs, n_categs)
    n_confuson = 1000

    # confusion matrix of 18 X 18 size
    for i in range(n_confuson):
        category, name, category_tensor, name_tensor = randomTrainingExample()
        # Generate random samples
        output = evaluate(name_tensor)  # get the output from model
        guess, guess_i = categFromOutput(output)  # get category from output
        category_i = all_categs.index(category)  # get category index
        confusion[category_i][guess_i] += 1  # update confusion matrix value

    # norm the confusion matrix
    for i in range(n_categs):
        confusion[i] = confusion[i] / confusion[i].sum()

    # create plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # setup axes
    ax.set_xticklabels([''] + all_categs, rotation=90)
    ax.set_yticklabels([''] + all_categs)

    # force label in each tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # show the plot
    plt.show()
    
    predict('Dovesky')
    predict('Lucas')
    predict('Sundaram')

