import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Sample paragraph (changed)
paragraph = """Natural Language Processing enables computers to understand human language.
Google Translate, Siri, and chatbots use NLP to process text and speech effectively."""

# 1️⃣ Tokenize into sentences
sentences = nltk.sent_tokenize(paragraph)

lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    # Lemmatize and remove stopwords
    words = [
        lemmatizer.lemmatize(word.lower(), pos='v')
        for word in words
        if word.lower() not in set(stopwords.words('english'))
    ]
    sentences[i] = ' '.join(words)

print("After removing stopwords and lemmatization:\n", sentences)

# 2️⃣ POS tagging
print("\nPart-of-Speech (POS) Tags:")
for sentence in nltk.sent_tokenize(paragraph):
    words = nltk.word_tokenize(sentence)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    pos_tags = nltk.pos_tag(words)
    print(pos_tags)

# 3️⃣ Named Entity Recognition
sentence = "Elon Musk founded SpaceX in 2002 in California."
words = nltk.word_tokenize(sentence)
tags = nltk.pos_tag(words)
print("\nNamed Entity Recognition:")
nltk.ne_chunk(tags).pprint()
