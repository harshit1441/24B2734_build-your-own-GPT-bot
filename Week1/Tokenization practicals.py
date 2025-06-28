import nltk

# Download punkt tokenizer (only needs to be run once)
nltk.download('punkt')

# Text to tokenize
corpus = """My name is Samiksha.
I'm learning this for SOC!"""

# Import tokenizers
from nltk.tokenize import sent_tokenize, word_tokenize

# Sentence tokenization
documents = sent_tokenize(corpus)
print("Sentences:", documents)

# Word tokenization (whole paragraph)
print("Words (whole corpus):", word_tokenize(corpus))

# Word tokenization per sentence
for sentence in documents:
    print("Words in sentence:", word_tokenize(sentence))

from nltk.tokenize import wordpunct_tokenize
print(wordpunct_tokenize(corpus))



