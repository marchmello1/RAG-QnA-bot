from langchain_community.document_loaders import PDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["streamlit"]["openai_api_key"]

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        text += PDFLoader(pdf).load()
    return text

def get_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_text(raw_text)
    return documents

def get_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def main():
    st.set_page_config(page_title="Picostone QnA bot", page_icon=":robot_face:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.markdown("<h1 style='text-align: center; color: #075E54;'>Picostone QnA Bot</h1>", unsafe_allow_html=True)
    question = st.text_input("Ask a question")
    
    if question:
        handle_question(question, openai_api_key)  # Pass the API key here
    else:
        st.warning("Type a question to start the conversation.")
    
    with st.sidebar:
        st.subheader("Upload Documents")
        docs = st.file_uploader("Upload PDF documents", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            with st.spinner("Processing"):
                if docs:
                    # Get the pdf
                    raw_text = get_pdf_text(docs)
                    
                    # Get the text chunks
                    documents = get_chunks(raw_text)
                    
                    # Create vectorstore
                    vectorstore = get_vectorstore(documents)
                    
                    # Define LLM
                    llm = ChatOpenAI(openai_api_key=openai_api_key)

                    # Define prompt template
                    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

                    <context>
                    {context}
                    </context>

                    Question: {input}""")
                    
                    # Create a retrieval chain to answer questions
                    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), llm, prompt)
                    
                    # Store conversation chain in session state
                    st.session_state.conversation = retrieval_chain
                else:
                    st.warning("No PDF files uploaded. Continuing conversation without searching from PDFs.")

if __name__ == '__main__':
    main()
