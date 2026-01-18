import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample training data - balanced between real (0) and fake (1) news
training_data = {
    'text': [
        # Real news examples (label 0)
        'The stock market rose today as investors gained confidence. Major indices including the S&P 500 showed gains across most sectors.',
        'Scientists announced a breakthrough in renewable energy technology. The new solar panels have 40% higher efficiency than previous models.',
        'The city council approved a new public transportation plan. The initiative will add 50 new buses to reduce traffic congestion.',
        'Health officials reported a decrease in flu cases this winter. Vaccination rates reached record levels in most communities.',
        'Congressional Democrats introduced new education funding bill. The legislation aims to improve schools in low-income areas.',
        'The Federal Reserve announced interest rate increase. Officials cited inflation concerns as the main reason for the decision.',
        'Tech company releases new smartphone with advanced features. The device offers improved battery life and processing power.',
        'Environmental agency reports air quality improvements. Pollution levels decreased by 15% compared to last year.',
        'University research team discovers new medical treatment. Clinical trials show promising results for disease prevention.',
        'Global trade agreement reaches consensus among nations. Negotiations took three years to complete successfully.',
        
        # Fake news examples (label 1)
        'SHOCKING: Secret cure for all diseases hidden by government! Big pharma does not want you to know this truth.',
        'Aliens confirmed living under the earth! NASA has been covering this up for decades. Anonymous sources reveal the conspiracy.',
        'Famous celebrity dies in car crash - but is actually still alive! Hospitals and government faking deaths.',
        'New world order being established right now! World leaders secretly meeting to control global population.',
        'Miracle weight loss pill discovered - lose 50 pounds in one week! Doctors hate this simple trick.',
        '5G towers spreading deadly virus - scientists confirm! Government forcing 5G rollout to harm population.',
        'Proof that moon landing was fake! Leaked footage shows Hollywood studio used to film the hoax.',
        'Politician caught in massive scandal - mainstream media covering it up! Underground internet sources have the real story.',
        'Ancient civilization found with advanced technology! Experts kept secret to control historical narrative.',
        'Chemical in water supply causing mind control! Source says fluoride making people obedient to government.'
    ],
    'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(training_data)

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
