import time
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
from io import BytesIO
import PyPDF2

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = "medical-chatbot"
DEFAULT_FILE = "medical_book.pdf"

@st.cache_resource
def initialize_pinecone():
    """Initialize Pinecone and connect to or create the index."""
    pinecone_instance = Pinecone(api_key=PINECONE_API_KEY)
    serverless_spec = ServerlessSpec(region=PINECONE_ENVIRONMENT, cloud="was")
    existing_indexes = [index['name'] for index in pinecone_instance.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pinecone_instance.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=serverless_spec
        )
    return pinecone_instance.Index(PINECONE_INDEX_NAME)

def process_pdf(file_bytes):
    """Extract text from a PDF file and split it into chunks."""
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text.strip() + " "  # Add a space between pages to maintain continuity

    if not text.strip():
        raise ValueError("The uploaded PDF does not contain any readable text.")
    
    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    documents = text_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in documents]

def process_url(url):
    """Extract text content from a URL."""
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve URL. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])  # Extract text from <p> tags
    
    if not text.strip():
        raise ValueError("The URL does not contain any readable text.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    documents = text_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in documents]

@st.cache_data
def upload_to_pinecone(_texts, _index):
    """Generate embeddings and upload them to Pinecone."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    PineconeVectorStore.from_documents(
        documents=_texts,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace="uploaded_url"
    )

def query_chatbot(question, namespace="uploaded_pdf"):
    """Query the chatbot using the provided question."""
    # Check if the Ollama server is reachable
    try:
        requests.get("http://localhost:11434", timeout=5)
    except requests.ConnectionError:
        raise ConnectionError("Ollama server is not running. Please start the Ollama server with `ollama serve`.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    llm = Ollama(model="llama3.2")  # Ensure Ollama is running locally
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"namespace": namespace}),
        return_source_documents=True
    )
    
    result = qa_chain({"query": question})
    response = result.get("result", "").strip()
    source_docs = result.get("source_documents", [])
    
    if not response:
        return "I don't know.", None

    # Collect source information
    source_info = "\n\n".join(
        [f"Source: Page {doc.metadata.get('page_number', 'Unknown')}\n{doc.page_content[:200]}..." for doc in source_docs]
    )
    return response, source_info

# Streamlit app configuration
st.set_page_config(page_title="Healthcare Chatbot", layout="wide", page_icon="🩺")

st.title("Healthcare Chatbot")
st.subheader("Your virtual assistant for healthcare-related queries.")

# Input handling and processing
input_type = st.sidebar.selectbox("Select Input Type:", ["PDF", "URL", "Default"], index=0)
file_uploaded = st.sidebar.file_uploader("Upload a PDF", type="pdf") if input_type == "PDF" else None
url_input = st.sidebar.text_input("Enter a URL") if input_type == "URL" else None
query = st.text_input("Ask your healthcare question:", placeholder="E.g., What is the treatment for hypertension?")
submit_button = st.button("Submit Question")

if submit_button:
    if query:
        try:
            _index = initialize_pinecone()

            if input_type == "PDF" and file_uploaded:
                file_bytes = file_uploaded.read()
                _texts = process_pdf(file_bytes)
                upload_to_pinecone(_texts, _index)
            elif input_type == "URL" and url_input:
                _texts = process_url(url_input)
                upload_to_pinecone(_texts, _index)
            elif input_type == "Default":
                if not os.path.exists(DEFAULT_FILE):
                    st.error("Default file not found. Please upload a PDF.")
                    st.stop()
                with open(DEFAULT_FILE, "rb") as f:
                    file_bytes = f.read()
                    _texts = process_pdf(file_bytes)
                    upload_to_pinecone(_texts, _index)
            else:
                st.error("Please provide a valid input.")
                st.stop()

            response, sources = query_chatbot(query)
            st.markdown("### Chatbot's Response:")
            st.write(response)
            if sources:
                st.markdown("### Sources:")
                st.write(sources)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a question to query the chatbot.")
