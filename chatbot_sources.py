# chatbot.py
import streamlit as st
import fitz  # PyMuPDF for PDF parsing
from llama_index.core import Document
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_ollama.llms import OllamaLLM
import os

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Set up the Streamlit framework
st.title('Langchain Chatbot With Llama2 Model and PDF Integration')
input_text = st.text_input("Ask your question!")

# Initialize the Ollama model
llm = OllamaLLM(model="llama2")

# LlamaIndex Setup
####### Path to folder here!!!!!!################################################
pdf_folder = r'C:\Users\ida\OneDrive\Skrivbord\TRA235\Docs_1'
documents = []

# Read and process PDF documents
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        pdf_text = extract_text_from_pdf(pdf_path)
        document = Document(
            text=pdf_text,
            id=filename,
            title=filename
        )
        documents.append(document)

# Debug: Check documents
st.write(f"Total Documents Indexed: {len(documents)}")
st.write(f"Sample Document: {documents[0].text[:500] if documents else 'No documents found'}")

# Create embedding model using HuggingFace embeddings
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create index and query engine
index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
query_engine = index.as_query_engine(llm=llm)

# Function to query the index
def query_llama_index(question):
    st.write(f"Querying index with question: {question}")
    try:
        response = query_engine.query(question)
        st.write(f"Raw Query Response: {response}")
        st.write(f"Response Text: {response.response}")
        return response.response
    except Exception as e:
        st.write(f"Error querying index: {e}")
        return "An error occurred while querying the index."

# Process user input
if input_text.strip():
    st.write("Processing the question...")
    st.write(f"Input Received: {input_text}")

    # Query the index
    index_response = query_llama_index(input_text)
    st.write(f"Indexed Information Response: {index_response}")
else:
    st.write("No valid input provided.")
