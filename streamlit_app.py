import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np

st.title("Fake News Detection")

st.info("⚠️ Note: This model was trained on a specific dataset. It works best with political and general news articles. Results may not be 100% accurate.")

# session state for input
if 'news_text' not in st.session_state:
    st.session_state['news_text'] = ""

news_text = st.text_area("Enter news text to check:", value=st.session_state['news_text'], key='news_text', help="Best results with political and general news articles")

# Load vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("finalized_model.pkl", "rb") as f:
    model = pickle.load(f)

def load_csv(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                return pd.read_csv(p)
        except Exception:
            continue
    return None

csv_paths = [
    "news-checkpoint.csv",
    os.path.join(".ipynb_checkpoints", "news-checkpoint.csv"),
    os.path.join("..", ".ipynb_checkpoints", "news-checkpoint.csv"),
]
df = load_csv(csv_paths)

# Example selector and ground-truth display
st.markdown("---")
st.header("Example (ground-truth available)")
selected_row = None
if df is None:
    st.info("Dataset not found in workspace. Place `news-checkpoint.csv` in project root or .ipynb_checkpoints to enable examples.")
else:
    if {'title','text','label'}.issubset(df.columns):
        real_examples = df[df['label'].astype(str).str.upper()=='REAL'][['title','text','label']].head(20).to_dict(orient='records')
        if len(real_examples) == 0:
            st.write("No REAL-labeled examples found in CSV.")
        else:
            titles = [e['title'] for e in real_examples]
            sel = st.selectbox("Select an example to preview:", titles)
            selected_row = next(e for e in real_examples if e['title']==sel)
            st.write("**Title:**", selected_row['title'])
            st.write(selected_row['text'][:800] + ("..." if len(selected_row['text'])>800 else ""))
            st.write("**Ground-truth label:**", selected_row['label'])
            if st.button("Load example into input"):
                st.session_state['news_text'] = selected_row['text']
                st.experimental_rerun()
    else:
        st.write("CSV missing required columns `title`, `text`, or `label`.")

st.markdown("---")
demo_mode = st.checkbox("Demo mode: show CSV ground-truth for selected examples", value=False, help="When enabled, the app will prominently show the CSV label for loaded examples (useful for presentations).")

if st.button("Check News"):
    text_to_check = st.session_state.get('news_text', '').strip()
    if text_to_check == "":
        st.warning("Please enter some text!")
    else:
        X = vectorizer.transform([text_to_check])
        pred_raw = model.predict(X)[0]

        # Robust mapping: handle models that output labels or integer indices
        label = None
        if hasattr(model, 'classes_'):
            classes = list(getattr(model, 'classes_'))
        else:
            classes = None

        if isinstance(pred_raw, (int, np.integer)):
            if classes:
                label = classes[int(pred_raw)]
            else:
                label = str(pred_raw)
        else:
            label = str(pred_raw)

        # show model result
        if label.upper() == 'REAL':
            st.success(f"✅ Model prediction: {label}")
        else:
            st.error(f"❌ Model prediction: {label}")

        # Demo mode or ground-truth display
        shown_gt = False
        if demo_mode and selected_row is not None and text_to_check == selected_row['text']:
            st.info(f"Ground-truth label (demo mode): {selected_row['label']}")
            shown_gt = True
        else:
            if selected_row is not None and text_to_check == selected_row['text']:
                st.info(f"Ground-truth label: {selected_row['label']}")
                shown_gt = True
            else:
                if df is not None:
                    match = df[df['text'].astype(str).str.strip() == text_to_check]
                    if not match.empty:
                        gt = match.iloc[0]['label']
                        st.info(f"Ground-truth label found in CSV: {gt}")
                        shown_gt = True

        if not shown_gt and demo_mode:
            st.info("Demo mode is enabled but no matching example is loaded. Load an example to show its CSV label.")

