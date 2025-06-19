from data.fetch_data import fetch_spam_data
from preprocessing.email_loader import load_emails
from pipeline.pipeline import build_pipeline
from model.train import train_model
from model.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Download data
ham_dir, spam_dir = fetch_spam_data()
ham_emails = load_emails(ham_dir)
spam_emails = load_emails(spam_dir)

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0]*len(ham_emails) + [1]*len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Preprocess and train
pipeline = build_pipeline(vocab_size=1000)
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

model = train_model(X_train_transformed, y_train)
evaluate_model(model, X_test_transformed, y_test)
