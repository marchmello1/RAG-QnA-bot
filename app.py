import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

# Load environment variables from .env file
load_dotenv()

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["streamlit"]["openai_api_key"]

# Custom prompt template for rephrasing follow-up questions
custom_template = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:
"""

# Prompt template instance
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            text += "\n" 
    return text

# Convert text to chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)   
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Generate vector store
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device':'cpu'})
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Generate conversation chain  
def get_conversationchain(vectorstore, openai_api_key):
    llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') 
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory)
    return conversation_chain

# Generate response from user queries and display them accordingly
def handle_question(question, openai_api_key):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': question})
        if response["answer"]:
            st.session_state.chat_history = response["chat_history"]
            for i, msg in enumerate(reversed(st.session_state.chat_history)):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
            return

    if st.session_state.conversation and response.get("answer", "").startswith("I don't know"):
        llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
        response = llm.predict(question)
        st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
        return

    llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
    response = llm.predict(question)
    st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Picostone QnA bot", page_icon=":robot_face:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.markdown("<h1 style='text-align: center; color: #075E54;'>QnA Bot</h1>", unsafe_allow_html=True)
    question = st.text_input("Ask a question")
    
    if question:
        handle_question(question, openai_api_key)  
    else:
        st.warning("Type a question to start the conversation.")
    
    with st.sidebar:
        st.subheader("Upload Documents")
        docs = st.file_uploader("Upload PDF documents", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            with st.spinner("Processing"):
                if docs:
                    raw_text = get_pdf_text(docs)
                    text_chunks = get_chunks(raw_text)
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversationchain(st.session_state.vectorstore, openai_api_key)  
                else:
                    st.warning("No PDF files uploaded. Continuing conversation without searching from PDFs.")

if __name__ == '__main__':
    main()
