
# 📘 Build Your Own GPT

**Student:** Harshit Agarwal 
**Roll No.:** 24B2734  
**Project UID:** 143

## 🧠 Project Overview

This project is an exploration of Natural Language Processing (NLP) using **Python** and **PyTorch**, focusing on both foundational and advanced concepts. From traditional text preprocessing and vectorization techniques to deep learning models like **Word2Vec**, **LSTM**, and **BERT**, this journey has been rooted in both theory and hands-on implementation. The end goal was to build a strong understanding of NLP pipelines and apply them to real-world problems like semantic similarity and sentiment analysis.

---

## 🔍 Key Learnings

### 1. 🧹 Text Cleaning and Preprocessing
- Tokenization
- Removal of stopwords
- Stemming (Porter, Snowball)
- Lemmatization with WordNet
- POS tagging and Named Entity Recognition (NER)

### 2. 🧾 Traditional Vectorization Techniques
- One-Hot Encoding
- Bag of Words (BoW – both binary and count-based)
- N-gram modeling (unigrams, bigrams, trigrams)
- TF-IDF for term relevance

### 3. 🌐 Word Embeddings and Semantics
- Trained custom **Word2Vec** models (CBOW and Skip-gram)
- Sentence embeddings via averaged word vectors
- Used **cosine similarity** to compute semantic closeness

### 4. 📖 Vocabulary and Text Structures
- Understanding the hierarchy: Corpus → Documents → Sentences → Words
- Differentiated between tokens, vocabulary, and semantic content

### 5. 🔁 Stemming vs Lemmatization
- Analyzed practical pros and cons
- Preferred Snowball stemmer for better linguistic handling over Porter

### 6. 🛠️ Libraries in NLP
- Compared **NLTK** (academic, modular) and **spaCy** (industrial-grade, efficient)

### 7. 🚫 Limitations of Traditional Techniques
- Sparse matrix representation
- Lack of semantic depth
- Inflexibility with unknown words
- Input length constraints

### 8. 🔍 Introduction to Neural NLP
- Basics of **RNN**, **LSTM**, and **GRU** models
- Introduction to Transformer models like **BERT** for sequence modeling

### 9. 🧠 Neural Networks
- Learned structure of ANN: input, hidden, output layers
- Concepts of weights, bias, activation functions
- Backpropagation and optimization (SGD, Adam)
- Implemented basic networks using One-Hot inputs and simple Word2Vec

---

## 🔧 PyTorch Fundamentals

### 10. 📦 Tensors and Operations
- Created and manipulated PyTorch tensors
- Performed slicing, reshaping, arithmetic, and matrix ops
- Used `requires_grad` to track computations for backpropagation
- Integrated with NumPy arrays

### 11. 📁 Dataset & Dataloader
- Created custom datasets with `torch.utils.data.Dataset`
- Batched data efficiently using `DataLoader`
- Iterated over batches using loops for training

### 12. 🏗️ Basic Training Loop
- Manual implementation of forward, backward, and optimizer steps
- Tracked loss and accuracy across epochs

### 13. 🧩 Modular Training with nn.Module
- Defined custom models using `nn.Module` and `nn.Sequential`
- Structured networks with layers, activations, and dropout
- Applied appropriate loss functions (e.g., `nn.CrossEntropyLoss`)
- Managed training pipelines with `Dataset`, `DataLoader`, and optimizers

### 14. 🧥 Fashion MNIST Classification
- Built a feedforward neural network
- Preprocessed and normalized images
- Evaluated with accuracy metrics

### 15. 📚 RNN-Based QA System
- Implemented question answering using RNN/LSTM layers
- Handled sequential data and padded variable-length inputs
- Trained on basic QA-style tasks with hidden state management

---

## ✅ Milestones Achieved

### 🎯 Milestone 1: Semantic Similarity Retrieval System
Developed a model to find the **Top 5 similar headlines** to a given query using **Word2Vec embeddings**.

**Workflow:**
- Loaded a dataset of 10,000 news headlines (JSON format)
- Preprocessed text using NLTK: tokenization, stopword removal, lowercasing, and lemmatization
- Trained a **Word2Vec** model using Gensim
- Created sentence vectors by averaging word embeddings
- Used **cosine similarity** to rank the top 5 closest sentences
- Tools: `NLTK`, `Gensim`, `NumPy`, `scikit-learn`, Google Colab

### 🎯 Milestone 2+3: Sentiment Classification with BERT + LSTM
Created a binary **sentiment analysis model** using **BERT embeddings** and a custom **LSTM-based classifier**.

**Workflow:**
- Dataset: 10,000 labeled movie reviews (split 80:20 train/test)
- Tokenized reviews using `bert-base-uncased` from Hugging Face
- Extracted frozen BERT embeddings for each sentence
- Created a `MovieReviewDataset` class for input preparation
- Model Architecture:
  - BERT embedding input
  - LSTM layer for sequence processing
  - Dropout for regularization
  - Fully connected layer for binary output
- Training:
  - Loss: `BCELoss` (binary cross-entropy)
  - Optimizer: `Adam`
  - Device-aware training with CUDA
  - Tracked loss and accuracy across epochs
  - Visualized results using `matplotlib`

---

## 🚀 Final Takeaways

By combining traditional NLP concepts with deep learning models, this project helped me gain:
- A structured understanding of how raw text is transformed into models
- Experience with building models from scratch in PyTorch
- Hands-on application of **BERT** and **LSTM** in practical tasks
- The ability to analyze model performance and optimize pipelines

---

## 🛠️ Tech Stack & Libraries Used
- Python, PyTorch, Gensim, scikit-learn, NumPy, Matplotlib
- Hugging Face Transformers, NLTK, Google Colab
