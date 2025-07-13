import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import base64
from dotenv import load_dotenv

def set_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_img, unsafe_allow_html=True)

set_bg_from_local("background.jpg")

st.markdown("""
    <style>
    textarea {
        background-color: rgba(0, 0, 0, 0.4) !important;
        color: white !important;
        border-radius: 10px;
        font-size: 16px;
    }

    div.stButton > button {
        background-color: #cc0000aa;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 20px;
        border: 1px solid #ff4444;
    }

    div.stButton {
        display: flex;
        justify-content: center;
    }

    .element-container .stAlert {
        background-color: rgba(0, 0, 0, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }

    .stAlert p {
        color: white !important;
    }

    .stInfo {
        background-color: rgba(25, 25, 25, 0.75) !important;
        padding: 15px;
        border-radius: 10px;
        color: white !important;
        font-size: 15px;
    }

    .block-container {
        padding-bottom: 100px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load logo ---

st.markdown("<h1 style='text-align: center; color: white;'>Fake News Classification App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Detect fake news using Machine Learning</p>", unsafe_allow_html=True)



port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

from google import genai

# --- Setup Gemini ---
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_fact_check(news, include_sources=True):
    prompt = f"""
    Consider the following news text:

    "{news}"

    Is this information accurate or fake?{" Please cite credible sources if it is accurate." if include_sources else ""}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        return response.text
    except Exception as e:
        return f" Gemini API Error: {e}"
    



if __name__ == '__main__':
    
    st.subheader("Input the News content below")

    sentence = st.text_area("Enter your news content here", "", height=200)
    predict_btt = st.button("predict")

    if predict_btt:
        if sentence.strip() == "":
            st.warning("Please enter some news content.")
        else:
            prediction_class = fake_news(sentence)

            if prediction_class == [1]:
                st.success('This news is likely **Reliable**')
                with st.spinner("Searching for sources..."):
                    sources = gemini_fact_check(sentence, include_sources=True)
                    st.markdown("Gathered Info from other sources:")
                    st.markdown(f"<div class='stInfo'>{reasoning}</div>", unsafe_allow_html=True)

            elif prediction_class == [0]:
                st.warning('This news may be **Unreliable**')
                with st.spinner("Cross-checking with other news platforms..."):
                    reasoning = gemini_fact_check(sentence, include_sources=False)
                    st.markdown("Information:")
                    st.markdown(f"<div class='stInfo'>{reasoning}</div>", unsafe_allow_html=True)