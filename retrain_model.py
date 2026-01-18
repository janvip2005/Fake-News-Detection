import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd

# Load training data from CSV file
df = pd.read_csv('.ipynb_checkpoints/news-checkpoint.csv')

print(f"Loading {len(df)} news articles...")
print(f"Real news: {(df['label'] == 0).sum()}")
print(f"Fake news: {(df['label'] == 1).sum()}")

if len(df) == 0:
    print("⚠️ No training data found!")
else:
    # Use only text column for training
    X_text = df['text'].fillna('').astype(str)
    y = df['label']
    
    # Vectorize the text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=1, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_text)

    # Train the model
    print("Training model...")
    model = PassiveAggressiveClassifier(max_iter=50, random_state=0)
    model.fit(X, y)

    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    with open('finalized_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("✅ Model retrained successfully!")
    print(f"Accuracy: {model.score(X, y):.2%}")
    print(f"Total training samples: {len(df)}")
