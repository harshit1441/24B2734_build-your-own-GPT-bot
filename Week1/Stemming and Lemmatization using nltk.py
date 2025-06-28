#%%

#PORTERSTEMMER

words=["eating","eats","eaten","writes","writing","programming","programs","history","finally","finalize"]
from nltk.stem import PorterStemmer
stemming=PorterStemmer()
for word in words:
    print(word,"---->",stemming.stem(word))
#history-->histori !!, similar thing happnens with 'congratulations'


#%%

#REGEXPSTEMMER

from nltk.stem import RegexpStemmer
reg_stemmer = RegexpStemmer('ing$|able$|s$|e$', min=4)
#the $ in the end denotes the ing will be removed if its there in the end, if no $,  will be removed from all parts of the word
reg_stemmer.stem('eating')


# %%

#SNOWBALL STEMMER

from nltk.stem import SnowballStemmer
snowballstemmer=SnowballStemmer('english')
for word in words:
    print(word,'---->', snowballstemmer.stem(word))

#%%

#LEMMATIZATION

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
#pos- post tage, v-verb, n-noun, a-adjective, r-adverb
lemmatizer.lemmatize('going', pos='v') 
for word in words:
    print(word, '----->', lemmatizer.lemmatize(word, pos='v'))


# %%
