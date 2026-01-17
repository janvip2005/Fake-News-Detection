import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load and prepare data the same way as in app.py
df = pd.read_csv("news.csv.csv")
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(df["text"], labels, test_size=0.2, random_state=20)

# Initialize and fit the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
vectorizer.fit(x_train)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("vectorizer.pkl created successfully!")
