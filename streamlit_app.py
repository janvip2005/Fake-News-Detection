import streamlit as st
import pickle

st.title("Fake News Detection")

news_text = st.text_area("Enter news text to check:", help="Best results with political and general news articles")

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load model
with open("finalized_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example news buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("📝 Try Example 1", type="secondary"):
        st.session_state.example_text = "The United Nations Security Council voted to increase humanitarian aid to developing countries. The resolution passed with support from all permanent members. Aid will focus on healthcare, education, and infrastructure development."
        st.rerun()
        
with col2:
    if st.button("📝 Try Example 2", type="secondary"):
        st.session_state.example_text = "Breaking: A new miracle pill discovered cures all diseases instantly. Doctors are shocked and angry about this breakthrough. Hospitals will go out of business. Big pharma is trying to hide this from the public."
        st.rerun()

# Fill text area with example if clicked
if 'example_text' in st.session_state and st.session_state.example_text:
    news_text = st.text_area("Enter news text to check:", value=st.session_state.example_text, help="Best results with political and general news articles", key="text_input")
    st.session_state.example_text = None

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

