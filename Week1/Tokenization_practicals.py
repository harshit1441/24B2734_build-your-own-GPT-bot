import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize

# Download tokenizer (only once)
nltk.download('punkt')

# Sample text (changed from original)
text = """Artificial Intelligence is transforming industries.
IIT Bombay students are building amazing projects during SOC!"""

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentence Tokenization:", sentences)

# Word tokenization for the entire text
print("\nWord Tokenization (Full Text):", word_tokenize(text))

# Word tokenization per sentence
print("\nWord Tokenization (Per Sentence):")
for s in sentences:
    print(f"{s} -> {word_tokenize(s)}")

# WordPunct tokenization (splits punctuation separately)
print("\nWordPunct Tokenization:", wordpunct_tokenize(text))



