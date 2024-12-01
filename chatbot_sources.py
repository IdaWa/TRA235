# chatbot.py
import streamlit as st
import fitz  # PyMuPDF for PDF parsing
from llama_index.core import Document
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_ollama.llms import OllamaLLM
import os
import re
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
def preprocess_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Optional: remove headers/footers based on patterns
    # Example: text = re.sub(r'Page \d+', '', text)
    return text.strip()

def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        pdf_text = preprocess_text(extract_text_from_pdf(pdf_path))
        chunks = split_into_chunks(pdf_text)
        for i, chunk in enumerate(chunks):
            document = Document(
                text=chunk,
                id=f"{filename}_chunk_{i}",
                title=f"{filename} - Part {i+1}"
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
