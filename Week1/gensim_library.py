import gensim.downloader as api

# Load a smaller, faster pre-trained Word2Vec model
print("Loading Word2Vec model (small & fast for demo)...")
wv = api.load('glove-wiki-gigaword-50')  # ~70MB, fast to download

# 1️⃣ Similar words
print("\nWords similar to 'computer':")
for word, similarity in wv.most_similar('computer', topn=5):
    print(f"{word}: {similarity:.4f}")

# 2️⃣ Word similarity
similarity_score = wv.similarity("physics", "chemistry")
print(f"\nSimilarity between 'physics' and 'chemistry': {similarity_score:.4f}")

# 3️⃣ Word arithmetic (vector math)
vec = wv['king'] - wv['man'] + wv['woman']
print("\nKing - Man + Woman ≈")
for word, similarity in wv.most_similar([vec], topn=5):
    print(f"{word}: {similarity:.4f}")

print("\n✅ Demo complete! This shows how word embeddings understand meaning.")
