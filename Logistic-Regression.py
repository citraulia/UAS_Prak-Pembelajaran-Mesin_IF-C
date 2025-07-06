#%%
import pandas as pd
import re
import string
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import graphviz
import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("train.csv")
df.head()

# %%
print("Missing Values:\n", df.isnull().sum())
print("\nDistribusi Label:\n", df['label'].value_counts())

# %%
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['sms_clean'] = df['sms'].apply(preprocess_text)

# %%
for i in range(3):
    print(f"Original : {df['sms'][i]}")
    print(f"Cleaned  : {df['sms_clean'][i]}")
    print("-" * 50)

# %%
X = df['sms_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# %%
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# %%
y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)[:, 1]

print("Akurasi   :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred, pos_label=1))
print("Recall    :", recall_score(y_test, y_pred, pos_label=1))
print("F1 Score  :", f1_score(y_test, y_pred, pos_label=1))
print("AUC       :", roc_auc_score(y_test, y_proba))

# %%
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['ham', 'spam'],
            yticklabels=['ham', 'spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Logistic Regression")
plt.tight_layout()
plt.show()

# %%
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# %%
joblib.dump(model, 'logreg_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model & TF-IDF vectorizer berhasil disimpan.")

# %%
dot = graphviz.Digraph(format='png')
dot.attr(rankdir='LR', size='10')

steps = [
    ("Desain Eksperimen", "Input: Tujuan penelitian & rumusan masalah", "Proses: Menyusun skema eksperimen", "Output: Skema eksperimen"),
    ("Perancangan Dataset", "Input: Preferensi rasa pengguna", "Proses: Membuat dataset dummy", "Output: Dataset kategorikal"),
    ("Pra-Pemrosesan Data", "Input: Dataset kategorikal", "Proses: Encoding & split data", "Output: Dataset numerik latih & uji"),
    ("Implementasi Model", "Input: Data latih", "Proses: Melatih model DT & KNN", "Output: Model terlatih"),
    ("Evaluasi Model", "Input: Model & data uji", "Proses: Evaluasi dengan metrik klasifikasi", "Output: Hasil performa model"),
    ("Pengembangan Prototipe", "Input: Model terbaik", "Proses: Bangun antarmuka & integrasi model", "Output: Prototipe sistem rekomendasi"),
    ("Dokumentasi & Laporan", "Input: Seluruh hasil penelitian", "Proses: Susun dokumentasi & laporan", "Output: Laporan akhir")
]

for i, (step, input_txt, process_txt, output_txt) in enumerate(steps):
    dot.node(f"A{i}", input_txt, shape='box', style='filled', fillcolor='#E0F7FA')
    dot.node(f"B{i}", process_txt, shape='ellipse', style='filled', fillcolor='#FFF9C4')
    dot.node(f"C{i}", output_txt, shape='box', style='filled', fillcolor='#C8E6C9')

    dot.edge(f"A{i}", f"B{i}")
    dot.edge(f"B{i}", f"C{i}")
    if i < len(steps) - 1:
        dot.edge(f"C{i}", f"A{i+1}", constraint='true')

dot.render('IPO_flowchart', cleanup=False)
from IPython.display import Image
Image("IPO_flowchart.png")
