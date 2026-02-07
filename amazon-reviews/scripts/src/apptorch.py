import streamlit as st
import numpy as np
import pickle
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from nltk.tokenize import word_tokenize
import nltk
from models.neural_network_torch import NeuralNetworkTorch
import os

# --------------------------------------------------
# ENV
# --------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")

# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------
@st.cache_data
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

def is_special(text):
    rem = ''
    for i in text:
        rem += i if i.isalnum() else ' '
    return rem

def to_lower(text):
    return text.lower()

nltk.download('stopwords')
nltk.download('punkt')

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

# --------------------------------------------------
# PYTORCH MODEL (MUST MATCH TRAINING)
# --------------------------------------------------
class SentimentModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# --------------------------------------------------
# LOAD MODEL + VECTORIZER (ONCE)
# --------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")

    with open("cv.pkl", "rb") as f:
        cv = pickle.load(f)

    input_dim = len(cv.vocabulary_)

    model = NeuralNetworkTorch(input_dim)

    model.load_state_dict(
        torch.load("model_torch.pth", map_location=device)
    )

    model.eval()
    return model, cv


# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------
def main():
    st.title("Deep Learning Model Deployment (PyTorch)")

    review = st.text_input("Enter text, Type here...")

    if st.button("Predict"):
        f1 = clean(review)
        f2 = is_special(f1)
        f3 = to_lower(f2)
        f4 = rem_stopwords(f3)
        f5 = stem_txt(f4)

        model, cv = load_model()

        # Build BoW vector
        inp = np.zeros(len(cv.vocabulary_))
        for word, idx in cv.vocabulary_.items():
            inp[idx] = f5.count(word)

        x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            y_pred = model(x).item()

        st.write(f"Model output: `{y_pred:.4f}`")

        if y_pred >= 0.5:
            st.success("POSITIVO")
        else:
            st.error("NEGATIVO")

    else:
        st.write("Press the above button..")

if __name__ == "__main__":
    main()