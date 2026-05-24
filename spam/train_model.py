import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("spam_dataset.csv")

# clean
df.columns = df.columns.str.strip()
df = df.dropna()

# =========================
# 2. FIX LABELS
# =========================
df["Class"] = df["Class"].map({"ham": 0, "spam": 1})

# =========================
# 3. FEATURES + LABELS
# =========================
X = df["Message"]   # TEXT DATA
y = df["Class"]     # LABELS

# =========================
# 4. TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# =========================
# 5. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# =========================
# 6. MODEL TRAINING
# =========================
model = LogisticRegression()
model.fit(X_train, y_train)

# =========================
# 7. SAVE MODEL
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# =========================
# 8. ACCURACY
# =========================
print("Accuracy:", model.score(X_test, y_test))

print("✅ Model trained successfully!")