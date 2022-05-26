import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack
nltk.download('punkt')
nltk.download('stopwords')
", ".join(stopwords.words('english'))

@st.cache(allow_output_mutation=True)
def loader():
    scikit_log_reg=pickle.load(open('model.pkl','rb'))
    word_tokenize=pickle.load(open('tokenize.pkl','rb'))
    word_vectorizer=pickle.load(open('word_vectorizer.pkl','rb'))
    char_vectorizer=pickle.load(open('char_vectorizer.pkl','rb'))
    return scikit_log_reg, word_tokenize, word_vectorizer, char_vectorizer

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

if __name__ == '__main__':
    scikit_log_reg, word_tokenize, word_vectorizer, char_vectorizer = loader()
    data = st.text_input("Enter Your Sentence: ")
    if data:
        sentence = str(data)
        sentence = remove_stopwords(sentence)
        tokenized_input = word_tokenize(sentence)
        word_vec = word_vectorizer.transform([tokenized_input])
        char_vec = char_vectorizer.transform([tokenized_input])
        final_ip = hstack([word_vec, char_vec])
        input_formmated = scikit_log_reg.predict(final_ip)

        if input_formmated==0 :
            st.write('Sentence is Positive')
        else :
            st.write('Sentence is Negative')
    else:
        st.write('Enter a statement')
