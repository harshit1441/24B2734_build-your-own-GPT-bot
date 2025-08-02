from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer, WordNetLemmatizer

words = ["running", "flies", "studying", "happily", "organized", "universities"]

# 1️⃣ Porter Stemmer
print("Porter Stemmer:")
stemmer = PorterStemmer()
for word in words:
    print(word, "->", stemmer.stem(word))

# 2️⃣ Snowball Stemmer
print("\nSnowball Stemmer:")
snowball = SnowballStemmer("english")
for word in words:
    print(word, "->", snowball.stem(word))

# 3️⃣ Regexp Stemmer
print("\nRegexp Stemmer (remove 'ing'/'ies'):")
reg_stemmer = RegexpStemmer('ing$|ies$', min=4)
for word in words:
    print(word, "->", reg_stemmer.stem(word))

# 4️⃣ Lemmatization
print("\nLemmatization:")
lemmatizer = WordNetLemmatizer()
for word in words:
    print(word, "->", lemmatizer.lemmatize(word, pos='v'))
