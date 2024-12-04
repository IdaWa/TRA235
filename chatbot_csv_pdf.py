import streamlit as st
import fitz  # PyMuPDF for PDF parsing
import pandas as pd  # For handling CSV
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

# Preprocessing and chunking functions
def preprocess_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Load and process PDFs
def process_pdfs(pdf_folder):
    documents = []
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
    return documents

# Load and process CSV
def process_csv(file_path):
    st.write(f"Loading CSV data from: {file_path}")
    try:
        # Adjust for semicolon delimiter and quoted fields
        df = pd.read_csv(file_path, delimiter=';', quotechar='"', escapechar='\\')
        st.write(f"CSV Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Ensure required columns are present
        if 'Symptom' in df.columns and 'Possible Cause' in df.columns and 'Corrective Action' in df.columns:
            documents = []
            for idx, row in df.iterrows():
                symptom = row['Symptom']
                possible_cause = row['Possible Cause']
                corrective_action = row['Corrective Action']
                
                # Combine into a single chunk of text
                content = f"Symptom: {symptom}\nPossible Cause: {possible_cause}\nCorrective Action: {corrective_action}"
                
                # Preprocess and split into smaller chunks if needed
                chunks = split_into_chunks(content)
                for i, chunk in enumerate(chunks):
                    document = Document(
                        text=chunk,
                        id=f"csv_row_{idx}_chunk_{i}",
                        title=f"Row {idx} - Part {i+1}"
                    )
                    documents.append(document)
            st.write(f"Total CSV Documents Indexed: {len(documents)}")
            return documents
        else:
            st.write("Error: Required columns 'Symptom', 'Possible Cause', and 'Corrective Action' not found in CSV.")
            return []
    except Exception as e:
        st.write(f"Error loading CSV: {e}")
        return []

# Streamlit UI setup
st.title('Langchain Chatbot With Llama2 Model, PDF, and CSV Integration')
input_text = st.text_input("Ask your question!")

# PDF and CSV file paths
pdf_folder = r'C:\Users\ida\OneDrive\Skrivbord\TRA235\Docs_pdfs'
csv_file_path = r'C:\Users\ida\OneDrive\Skrivbord\TRA235\CSV_files\Autodoor - Troubleshooting - Haas Service Manual.csv'

# Process PDFs and CSVs
pdf_documents = process_pdfs(pdf_folder)
csv_documents = process_csv(csv_file_path)

# Combine all documents
all_documents = pdf_documents + csv_documents

# Debug: Display combined document stats
st.write(f"Total Combined Documents: {len(all_documents)}")
if all_documents:
    st.write(f"Sample Document: {all_documents[0].text[:500]}")

# Create embedding model using HuggingFace embeddings
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create index and query engine
index = VectorStoreIndex.from_documents(all_documents, embed_model=embedding_model)
llm = OllamaLLM(model="llama2")
query_engine = index.as_query_engine(llm=llm)

# Query function
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
