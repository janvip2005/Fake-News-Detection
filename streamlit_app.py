import streamlit as st
import pickle

st.title("Fake News Detection")

news_text = st.text_area("Enter news text to check:")

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
        st.success(f"The news is: **{result}**")

