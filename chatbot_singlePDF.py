# chatbot.py
import streamlit as st
import fitz  # PyMuPDF for PDF parsing
from llama_index.core import Document
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_ollama.llms import OllamaLLM
import re

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit setup
st.title('Langchain Chatbot with Llama2 and PDF Integration')

# Hardcoded PDF file path
pdf_path = r'C:\Users\ida\OneDrive\Skrivbord\TRA235\Docs_pdfs\Bar Feeder - Maintenance - Haas Service Manual.pdf'

# Input for the user question
input_text = st.text_input("Ask your question!")

# Initialize the LLM (Llama2)
llm = OllamaLLM(model="llama2")

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocess text by removing extra whitespace and optional cleaning.
    """
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

# Function to split text into smaller chunks
def split_into_chunks(text, chunk_size=1000, overlap=200):
    """
    Split text into chunks with overlap for better query matching.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Process the hardcoded PDF
try:
    # Extract and preprocess the PDF text
    pdf_text = preprocess_text(extract_text_from_pdf(pdf_path))
    st.write("PDF Text Extracted Successfully!")

    # Split the text into chunks
    chunks = split_into_chunks(pdf_text)
    documents = []
    for i, chunk in enumerate(chunks):
        document = Document(
            text=chunk,
            id=f"chunk_{i}",
            title=f"PDF Chunk {i + 1}"
        )
        documents.append(document)

    # Debugging: Display the number of documents and a sample chunk
    st.write(f"Total Chunks Created: {len(documents)}")
    st.write(f"Sample Chunk: {documents[0].text[:500] if documents else 'No chunks found'}")

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
            st.write(f"Response Text: {response.response}")
            return response.response
        except Exception as e:
            st.write(f"Error querying index: {e}")
            return "An error occurred while querying the index."

    # Process user question if provided
    if input_text.strip():
        st.write("Processing your question...")
        response = query_llama_index(input_text)
        st.write(f"Response: {response}")
    else:
        st.write("Please provide a question to ask.")
except Exception as e:
    st.write(f"Error processing PDF: {e}")
