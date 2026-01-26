import os
import pickle
import sys
import pandas as pd

candidates = [
    "finalized_model.pkl",
    os.path.join("models","finalized_model.pkl"),
]
vec_candidates = [
    "vectorizer.pkl",
    os.path.join("models","vectorizer.pkl"),
]

def find_file(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    return None

model_path = find_file(candidates)
vec_path = find_file(vec_candidates)

print('Model path:', model_path)
print('Vectorizer path:', vec_path)

if model_path is None:
    print('No model file found. Place finalized_model.pkl in project root.')
    sys.exit(2)

with open(model_path,'rb') as f:
    model = pickle.load(f)

print('\nModel type:', type(model))
print('Has attribute classes_:', hasattr(model, 'classes_'))
if hasattr(model, 'classes_'):
    try:
        print('classes_:', getattr(model, 'classes_'))
    except Exception as e:
        print('Error reading classes_:', e)

# load vectorizer if present
vectorizer = None
if vec_path:
    with open(vec_path,'rb') as f:
        vectorizer = pickle.load(f)
    print('\nLoaded external vectorizer:', type(vectorizer))
else:
    print('\nNo external vectorizer loaded. Model may include preprocessing pipeline.')

# pick a sample from news_real.csv
csv_candidates = [
    'news_real.csv',
    os.path.join('.ipynb_checkpoints','news-checkpoint.csv'),
    'news-checkpoint.csv'
]
csv_path = None
for p in csv_candidates:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print('\nNo CSV found to sample from. Place news_real.csv or news-checkpoint.csv in project root.')
    sys.exit(3)

print('\nUsing CSV sample:', csv_path)
df = pd.read_csv(csv_path)
if df.empty:
    print('CSV is empty')
    sys.exit(4)

# use first row as sample
row = df.iloc[0]
text = row.get('text') or row.get('title') or str(row.values)
print('\nSample title:', row.get('title'))
print('\nSample first 300 chars of text:\n', text[:300])

# prepare X
if vectorizer is not None:
    X = vectorizer.transform([text])
else:
    # try to use model's .predict on raw text if model is a pipeline
    try:
        pred = model.predict([text])
        print('\nModel.predict on raw text ->', pred)
    except Exception as e:
        print('\nCould not call model.predict on raw text:', e)
        print('Try providing an external vectorizer.')
    # if we reached here, exit
    sys.exit(0)

# now predict
try:
    pred = model.predict(X)
    print('\nModel.predict ->', pred)
except Exception as e:
    print('\nError during model.predict on vectorized input:', e)

# predict_proba if available
if hasattr(model, 'predict_proba'):
    try:
        proba = model.predict_proba(X)
        print('\nModel.predict_proba ->', proba)
    except Exception as e:
        print('\npredict_proba error:', e)

# if classes_ exists and pred is numeric or index-based, show mapping
if hasattr(model, 'classes_'):
    classes = list(getattr(model,'classes_'))
    try:
        # if pred values are indices (e.g., numpy ints) map to classes
        mapped = [classes[int(p)] if (isinstance(p,(int,)) or (hasattr(p,'__int__') and not isinstance(p,str))) else p for p in pred]
        print('\nMapped prediction ->', mapped)
    except Exception as e:
        print('\nCould not map predictions to classes:', e)

print('\nDone.')
