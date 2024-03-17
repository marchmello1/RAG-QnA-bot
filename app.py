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
            text += "\n"  # Add a newline character to separate text from different pages
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
    if not chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device':'cpu'})
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Generate conversation chain  
def get_conversationchain(vectorstore, openai_api_key, chat_history):
    if not vectorstore:
        return None
    
    llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') 
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory,
                                initial_memory=chat_history)  # Pass the initial conversation history here
    return conversation_chain

# Generate response from user queries and display them accordingly
def handle_question(question, conversation):
    if not conversation:
        return "Error: Conversation chain is not initialized."
    
    response = conversation({'question': question})
    if response["answer"]:
        return response["chat_history"]
    return "No response found."

def main():
    st.set_page_config(page_title="Picostone QnA bot", page_icon=":robot_face:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    conversation = None
    chat_history = None

    st.markdown("<h1 style='text-align: center; color: #075E54;'>Picostone QnA Bot</h1>", unsafe_allow_html=True)
    question = st.text_input("Ask a question")
    
    if question:
        response = handle_question(question, conversation)
        if isinstance(response, list):
            for i, msg in enumerate(response):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.error(response)
    
    with st.sidebar:
        st.subheader("Upload Documents")
        docs = st.file_uploader("Upload PDF documents", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if docs:
                raw_text = get_pdf_text(docs)
                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                conversation = get_conversationchain(vectorstore, openai_api_key, chat_history)
            else:
                st.warning("No PDF files uploaded. Continuing conversation without searching from PDFs.")

if __name__ == '__main__':
    main()
