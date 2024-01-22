import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


def create_corpus(raw_text: str):
    """Tokenizes, removes stopwords, stems and cleans spl characters"""
    token_text = nltk.word_tokenize(raw_text)
    # download the stop words for english using nltk.download first
    stp_word = set(stopwords.words('english'))  # get the set of stop_words
    # remove the stop words
    filter_text = [word for word in token_text if word not in stp_word]
    stemmer = PorterStemmer()
    # stem the words
    stem_text = [stemmer.stem(word) for word in filter_text]
    # remove spl chars
    nospl_text = [re.sub(r'[^a-zA-Z\s]', '', word) for word in stem_text]
    # remove spaces
    fin_text = [word for word in nospl_text if word != '']
    # send the final list of words
    return list(set(fin_text))


example = """Autograd is a reverse automatic differentiation system.
Conceptually, autograd records a graph recording all of the operations
that created the data as you execute operations, giving you a directed
acyclic graph whose leaves are the input tensors and roots are the
output tensors. By tracing this graph from roots to leaves,
you can automatically compute the gradients using the chain rule."""

print(len(create_corpus(example)))
# the returned list contains the words in random order, and order 
# changes every run, so checking only len of the list

assert len(create_corpus(example)) == 28
