#python -m streamlit run app.py 

import streamlit as st
from transformers import pipeline
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
from inference_script import answer_question #import function from another file
from corpus import corpusvalue
import numpy as np
import getpass
import os
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import pickle
import sqlite3

@st.cache_resource
def load_model():
    return BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)

def load_rerank_model():
    return FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

def initLLM():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAuKPswmbdM8jCpSt0luez7tjLND-uyY7M"
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    template = """
    คุณเป็นผู้เชี่ยวชาญด้านกฎหมายจราจร มีหน้าที่ในการนำข้อความทางกฎหมายเเละข้อปฎิบัติเกี่ยวกับการละเมิดกฎจราจรเเละข้อปฎิบัติต่างๆมาตอบคำถามว่าคำถามที่ถามมานั้นว่าผิดหรือไม่หรือจะต้องปฎิบัติตัวอย่างไร เเละอธิบายเพิ่มเติม ให้รายละเอียดและคำอธิบายเพิ่มเติมเพื่อให้ผู้ที่ไม่ใช่ผู้เชี่ยวชาญด้านกฎหมายเข้าใจได้ง่ายขึ้น

    นี้คือคำถาม : {question}
    ข้อความทางกฎหมาย: {section}

    คำอธิบายโดยละเอียด:
    """
    prompt = PromptTemplate(
        input_variables=["section","question"],
        template=template
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain
def embeded_corpus():
    file_path_embeded_corpus = "save/BGM3savesimilar_Corpus" #
    with open(file_path_embeded_corpus,'rb') as file :
        BGM3similar_Corpus = pickle.load(file)
    return BGM3similar_Corpus

def insert_feedback(question, answer,like,dislike, feedback_text):
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS qa_feedback
                 (id INTEGER PRIMARY KEY, question TEXT, answer TEXT, 
                 like INTEGER, dislike INTEGER, feedback_text TEXT)''')
    data_to_insert = (question, answer, like, dislike, feedback_text)
    sql_query = 'INSERT INTO qa_feedback (question, answer, like, dislike, feedback_text) VALUES (?, ?, ?, ?, ?)'
    cursor.execute(sql_query, data_to_insert)
    conn.commit()
    conn.close()

model = load_model()
rerank_model = load_rerank_model()
llm_chain = initLLM()
BGM3similar_Corpus = embeded_corpus()
corpus_list = corpusvalue()

st.title("Traffic Law Question-Answering")

question = st.text_area("Enter your question:")

if 'like_value' not in st.session_state:
    st.session_state.like_value = 0
if 'dislike_value' not in st.session_state:
    st.session_state.dislike_value = 0

if st.button("Get Answer"):
    if question:
        answer = answer_question(question=question,model=model,rerankmodel=rerank_model,corpus_embed= BGM3similar_Corpus, corpus_list=corpus_list,llm_chain=llm_chain)
        st.text_area("Answer:", value=answer, height=500)

        st.write("### Feedback")
        feedback = st.text_area("Your feedback:")
        like = st.button("👍 Like")
        dislike = st.button("👎 Dislike")

        like_value = 1 if like else 0
        dislike_value = -1 if dislike else 0
        feedback = feedback if feedback else "No Feed back"

        if like or dislike or feedback:
            insert_feedback(question, answer,  like_value, dislike_value,feedback)