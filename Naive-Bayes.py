#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# %%
data = pd.read_csv('train.csv')

data.head()

# %%
print("Jumlah duplikat:", data.duplicated().sum())
data = data.drop_duplicates()
print("Setelah hapus duplikat:", data.duplicated().sum())
print("Jumlah null:\n", data.isnull().sum())

# %%
data['label'].value_counts()

# %%
X = data['sms']
Y = data['label']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_vectorized, Y, test_size=0.2, random_state=42)

# %%
start_time = time.time()
model = MultinomialNB()
model.fit(x_train, y_train)
training_time = time.time() - start_time

# %%
y_predict = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:,1]

print("=== Built-in Model Evaluation ===")
print(f'Training time: {training_time:.4f} seconds')
print(f'Accuracy     : {accuracy_score(y_test, y_predict) * 100:.2f}%')
print(f'Precision    : {precision_score(y_test, y_predict):.2f}')
print(f'Recall       : {recall_score(y_test, y_predict):.2f}')
print(f'F1 Score     : {f1_score(y_test, y_predict):.2f}')
print(f'AUC Score    : {roc_auc_score(y_test, y_prob):.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_predict))

# %%
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Built-in Model")
plt.tight_layout()
plt.show()

# %%
def tokenize(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.split()

# %%
def train_multinomial_nb(X, Y):
    vocab = set()
    class_counts = defaultdict(int)
    word_counts = defaultdict(lambda: defaultdict(int))

    for text, label in zip(X, Y):
        words = tokenize(text)
        class_counts[label] += 1
        for word in words:
            word_counts[label][word] += 1
            vocab.add(word)

    vocab = list(vocab)
    class_probs = {label: count / len(Y) for label, count in class_counts.items()}
    
    word_probs = {}
    for label in class_counts:
        total_words = sum(word_counts[label].values())
        word_probs[label] = {
            word: (word_counts[label][word] + 1) / (total_words + len(vocab))
            for word in vocab
        }

    return class_probs, word_probs, vocab

# %%
def predict_multinomial_nb(X, class_probs, word_probs, vocab):
    predictions = []
    for text in X:
        words = tokenize(text)
        scores = {}
        for label in class_probs:
            score = math.log(class_probs[label])
            for word in words:
                if word in vocab:
                    score += math.log(word_probs[label].get(word, 1 / (len(vocab))))
            scores[label] = score
        predictions.append(max(scores, key=scores.get))
    return predictions

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=11)

class_probs, word_probs, vocab = train_multinomial_nb(X_train, Y_train)
Y_pred = predict_multinomial_nb(X_test, class_probs, word_probs, vocab)

print("=== From Scratch Model Evaluation ===")
print("Accuracy:", accuracy_score(Y_test, Y_pred)*100, "%")
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

# %%
