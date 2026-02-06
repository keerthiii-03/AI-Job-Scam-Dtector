import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Get absolute path of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Backend folder path
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

# Create backend folder if not exists
os.makedirs(BACKEND_DIR, exist_ok=True)

# Dataset
data = {
    "text": [
        "Pay registration fee to get job",
        "Official interview call from company HR",
        "No interview direct offer letter",
        "Work from home earn 50000 per month",
        "Job opening with interview and company email"
    ],
    "label": ["scam", "real", "scam", "scam", "real"]
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model files (ABSOLUTE PATH)
model_path = os.path.join(BACKEND_DIR, "model.pkl")
vectorizer_path = os.path.join(BACKEND_DIR, "vectorizer.pkl")

pickle.dump(model, open(model_path, "wb"))
pickle.dump(vectorizer, open(vectorizer_path, "wb"))

print("✅ Model trained and saved successfully")
