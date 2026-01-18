import streamlit as st
import pickle

st.title("Fake News Detection")

st.info("⚠️ Note: This model was trained on a specific dataset. It works best with political and general news articles. Results may not be 100% accurate.")

news_text = st.text_area("Enter news text to check:", help="Best results with political and general news articles")

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load model
with open("finalized_model.pkl", "rb") as f:
    model = pickle.load(f)

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        X = vectorizer.transform([news_text])
        prediction = model.predict(X)[0]
        
        # Reverse the prediction (0=REAL, 1=FAKE)
        result = "REAL" if prediction == 0 else "FAKE"
        
        if result == "REAL":
            st.success("✅ **REAL NEWS** - This appears to be authentic news")
        else:
            st.error("❌ **FAKE NEWS** - This may be false or misleading")

