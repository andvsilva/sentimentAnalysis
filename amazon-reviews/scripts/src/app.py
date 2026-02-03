import streamlit as st
import numpy as np
import pickle
from keras.models import model_from_json
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from nltk.tokenize import word_tokenize
import nltk

@st.cache_data
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
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
    ss = SnowballStemmer('english') #tem portugues também
    return " ".join([ss.stem(w) for w in text])
def print_result(result):
    text,analysis_result = result
    print_text = "Positive" if analysis_result[0]=='1' else "Negative"
    return text,print_text

json_file = open('modelkeras.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("modelkeras.weights.h5")

def main():
    st.title('Deep Learning Model Deployment')
    
    review = str(st.text_input("Enter text, Type here...", ""))
    
    if st.button("Predict"):
        with open('cv.pkl','rb') as f:
            cv = pickle.load(f)
            f11 = clean(review)
            f22 = is_special(f11)
            f33 = to_lower(f22)
            f44 = rem_stopwords(f33)
            f55 = stem_txt(f44)
            bow,words = [],word_tokenize(f55)
        for word in words:
            bow.append(words.count(word))
        word_dict = cv.vocabulary_
        inp3 = []
        for i in word_dict:
            inp3.append(f55.count(i[0]))
        
        size_inp3 = len(inp3)
                
        y_pred2 = model.predict(np.array(inp3).reshape(1,size_inp3))
    
        print(f'>>>>>  {y_pred2}')
        
        if y_pred2 > 0:
            a = 'POSITIVO'
        else:
            a = 'NEGATIVO'
        st.success(a)
    
    else:
        st.write("Press the above button..")
    
if __name__ == '__main__':
    main()