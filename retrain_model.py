import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd

# Load your training data from CSV file
# df = pd.read_csv('news.csv')  # Should have 'text' and 'label' columns

# For now, using empty training data structure
training_data = {
    'text': [],
    'label': []
}

df = pd.DataFrame(training_data)

if len(df) == 0:
    print("⚠️ No training data provided!")
    print("Add your news data with 'text' and 'label' columns (0=REAL, 1=FAKE)")
else:
    # Vectorize the text
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=1, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Train the model
    model = PassiveAggressiveClassifier(max_iter=50, random_state=0)
    model.fit(X, y)

    # Save the model and vectorizer
    with open('finalized_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("✅ Model retrained successfully!")
    print("Accuracy:", model.score(X, y))
    print(f"Total training samples: {len(df)}")
