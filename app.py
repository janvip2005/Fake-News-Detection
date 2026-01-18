from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('finalized_model.pkl', 'rb'))

# Load the fitted TF-IDF vectorizer used during training
with open("vectorizer.pkl", "rb") as f:
    vector = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    prediction_text = None
    confidence = None
    result_class = None
    if request.method == 'POST':
        news = request.form.get('news', '')
        
        if news:
            # Transform the input using the vectorizer
            news_vector = vector.transform([news])
            
            # Get prediction probabilities
            proba = model.decision_function(news_vector)[0]
            
            # Make prediction
            result = model.predict(news_vector)[0]
            
            # Calculate confidence (convert to percentage)
            confidence = abs(proba) * 10
            if confidence > 100:
                confidence = 100
            confidence = round(confidence, 1)
            
            # Reverse the prediction (0=REAL, 1=FAKE)
            final_result = "REAL" if result == 0 else "FAKE"
            result_class = "real" if final_result == "REAL" else "fake"
            prediction_text = final_result
    
    return render_template('prediction.html', prediction_text=prediction_text, confidence=confidence, result_class=result_class)
    
if __name__ == '__main__':
    app.run(debug=True)
