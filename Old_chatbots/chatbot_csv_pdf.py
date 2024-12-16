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

# Preprocessing and chunking functions for text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Load and process PDF documents
pdf_folder = r'C:\Users\ida\OneDrive\Skrivbord\TRA235\Docs_pdfs'
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

# Load and process the maintenance logs (CSV)
csv_file_path = r'C:\Users\ida\OneDrive\Skrivbord\TRA235\CSV_files\fake_maintenance_logs.csv'  # Assuming this is the path to the CSV
def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        documents = []
        for idx, row in df.iterrows():
            symptom = row['Symptom']
            possible_cause = row['Possible Cause']
            corrective_action = row['Corrective Action Taken']
            
            content = f"Symptom: {symptom}\nPossible Cause: {possible_cause}\nCorrective Action: {corrective_action}"
            chunks = split_into_chunks(content)
            for i, chunk in enumerate(chunks):
                document = Document(
                    text=chunk,
                    id=f"csv_row_{idx}_chunk_{i}",
                    title=f"Log Row {idx} - Part {i+1}"
                )
                documents.append(document)
        return documents
    except Exception as e:
        st.write(f"Error processing CSV: {e}")
        return []

csv_documents = process_csv(csv_file_path)

# Combine PDF and CSV documents
all_documents = documents + csv_documents

# Debug: Display combined document stats
st.write(f"Total Combined Documents: {len(all_documents)}")
if all_documents:
    st.write(f"Sample Document: {all_documents[0].text[:500]}")

# Initialize the Ollama model
llm = OllamaLLM(model="llama2")

# Create embedding model using HuggingFace embeddings
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create index and query engine for all documents
index = VectorStoreIndex.from_documents(all_documents, embed_model=embedding_model)
query_engine = index.as_query_engine(llm=llm)

# Function to query the index
def query_llama_index(question, source_type):
    st.write(f"Querying {source_type} with question: {question}")
    try:
        # Query the respective source (either PDF or CSV)
        if source_type == 'manual':
            response = query_engine.query(question)
        elif source_type == 'log':
            response = query_engine.query(question)
        st.write(f"Raw Query Response ({source_type}): {response}")
        st.write(f"Response Text ({source_type}): {response.response}")
        return response.response
    except Exception as e:
        st.write(f"Error querying {source_type}: {e}")
        return f"An error occurred while querying {source_type}."

# Streamlit UI setup
st.title('Langchain Chatbot With Llama2 Model, PDF and Maintenance Logs')
input_text = st.text_input("Ask your question!")

# Process user input
if input_text.strip():
    st.write("Processing the question...")
    st.write(f"Input Received: {input_text}")

    # Query the PDF documents (manuals)
    manual_response = query_llama_index(input_text, 'manual')

    # Query the CSV documents (maintenance logs)
    log_response = query_llama_index(input_text, 'log')

    # Display the results separately
    st.write("### Response from the Manual:")
    st.write(manual_response)
    
    st.write("### Response from the Maintenance Logs:")
    st.write(log_response)
else:
    st.write("No valid input provided.")
