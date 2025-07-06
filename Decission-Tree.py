#%%
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv("train.csv")
df.head()

# %%
print(df.info())

# %%
print("Jumlah duplikat:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplikat setelah dihapus:", df.duplicated().sum())
print("\nMissing value:\n", df.isnull().sum())

# %%
df['label'].value_counts()

# %%
X = df['sms']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# %%
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# %%
start_time = time.time()
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)
training_time = time.time() - start_time

# %%
y_pred = dt_model.predict(X_test_tfidf)
y_prob = dt_model.predict_proba(X_test_tfidf)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Training Time : {training_time:.4f} seconds")
print(f"Akurasi       : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-score      : {f1:.4f}")
print(f"AUC           : {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Decision Tree Model")
plt.tight_layout()
plt.show()
