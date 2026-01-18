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
        st.session_state.example_text = "President announces new economic stimulus package aimed at creating jobs. The package includes tax cuts and infrastructure investments. Economic advisors predict the package will boost GDP growth by 2-3 percent over the next fiscal year."
        st.rerun()
        
with col2:
    if st.button("📝 Try Example 2", type="secondary"):
        st.session_state.example_text = "Shocking discovery: Scientists claim they have found proof that the earth is flat. A group of researchers say NASA has been lying about the shape of our planet for decades. They claim to have video evidence but refuse to show it publicly."
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

