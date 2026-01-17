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
        X = vectorizer.transform([news_text])   # ✅ VERY IMPORTANT LINE
        prediction = model.predict(X)[0]        # ✅ NOT raw text
        st.success(f"The news is: **{prediction}**")

