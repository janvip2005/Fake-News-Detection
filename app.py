from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('finalized_model.pkl', 'rb'))

# We need to recreate the TfidfVectorizer with the same parameters
df = pd.read_csv("news.csv.csv")
labels = df.label
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df["text"], labels, test_size=0.2, random_state=20)

# Initialize and fit the vectorizer (same as in notebook)
vector = TfidfVectorizer(stop_words='english', max_df=0.7)
vector.fit(x_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    prediction_text = None
    if request.method == 'POST':
        news = request.form.get('news', '')
        
        if news:
            # Transform the input using the vectorizer
            news_vector = vector.transform([news])
            
            # Make prediction
            result = model.predict(news_vector)[0]
            prediction_text = f'News is: {result}'
    
    return render_template('prediction.html', prediction_text=prediction_text)
    
if __name__ == '__main__':
    app.run(debug=True)