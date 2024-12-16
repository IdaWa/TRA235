import streamlit as st
import fitz  # PyMuPDF for PDF parsing
import pandas as pd
from llama_index.core import Document
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_ollama.llms import OllamaLLM
import os
import re
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#####################################
# Configuration and File Paths
#####################################

pdf_folder = r'C:\Users\ida\OneDrive\Skrivbord\TRA235\Docs_pdfs'
csv_file_path = r'C:\Users\ida\OneDrive\Skrivbord\TRA235\CSV_files\fake_maintenance_logs.csv'

#####################################
# Functions
#####################################

def preprocess_text(text):
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-ASCII characters that can cause gibberish
    text = text.encode('ascii', errors='ignore').decode('ascii')
    return text.strip()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text
    return text

# Set up the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "?", "!"]
)

def process_pdfs(pdf_folder_path):
    pdf_documents = []
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, filename)
            pdf_text = preprocess_text(extract_text_from_pdf(pdf_path))
            chunks = text_splitter.split_text(pdf_text)

            for i, chunk in enumerate(chunks):
                doc = Document(
                    text=chunk,
                    doc_id=f"{filename}_chunk_{i}",
                    metadata={"source_file": filename, "doc_type": "manual"}
                )
                pdf_documents.append(doc)
    return pdf_documents

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        # We'll return both the docs and df so we can reference rows later
        csv_docs = []
        for idx, row in df.iterrows():
            symptom = row.get('Symptom', "")
            possible_cause = row.get('Possible Cause', "")
            corrective_action = row.get('Corrective Action Taken', "")

            content = f"Symptom: {symptom}\nPossible Cause: {possible_cause}\nCorrective Action: {corrective_action}"
            clean_content = preprocess_text(content)
            chunks = text_splitter.split_text(clean_content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    text=chunk,
                    doc_id=f"csv_row_{idx}_chunk_{i}",
                    metadata={
                        "source_type": "maintenance_log",
                        "row_id": idx,
                        "doc_type": "log",
                        "csv_file": os.path.basename(file_path)
                    }
                )
                csv_docs.append(doc)
        return csv_docs, df
    except Exception as e:
        st.write(f"Error processing CSV: {e}")
        return [], None

#####################################
# Main Code
#####################################

st.title('Langchain Chatbot With Llama2 Model, PDF and CSV Integration')

# Process Documents
pdf_documents = process_pdfs(pdf_folder)
csv_documents, csv_df = process_csv(csv_file_path)

st.write(f"Total Manual Documents: {len(pdf_documents)}")
st.write(f"Total Log Documents: {len(csv_documents)}")

if pdf_documents:
    st.write(f"Sample Manual Document: {pdf_documents[0].text[:500]}")
if csv_documents:
    st.write(f"Sample Log Document: {csv_documents[0].text[:500]}")

# Initialize LLM and embeddings
llm = OllamaLLM(model="llama2")
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create separate indexes
manual_index = VectorStoreIndex.from_documents(pdf_documents, embed_model=embedding_model)
log_index = VectorStoreIndex.from_documents(csv_documents, embed_model=embedding_model)

# Create query engines with additional context
manual_query_engine = manual_index.as_query_engine(
    llm=llm,
    additional_context=(
        "If any retrieved text is corrupted, garbled, or nonsensical, do not include it. "
        "Focus on providing a coherent, understandable, and helpful answer. "
        "Also, include relevant metadata from sources if possible."
    )
)

log_query_engine = log_index.as_query_engine(
    llm=llm,
    additional_context=(
        "If any retrieved text is corrupted, garbled, or nonsensical, do not include it. "
        "Focus on providing a coherent, understandable, and helpful answer. "
        "Also, include relevant metadata from sources if possible."
    )
)

def query_llama_index(question, source_type):
    logger.info(f"Querying {source_type} with question: {question}")
    st.write(f"### Querying {source_type.capitalize()} Index ###")
    st.write(f"**Question:** {question}")
    try:
        if source_type == 'manual':
            response = manual_query_engine.query(question)
        elif source_type == 'log':
            response = log_query_engine.query(question)
        else:
            st.write("Unknown source_type!")
            return None, None

        st.write("**Response Text:**", response.response)

        # Return both the response text and the source nodes for final combination
        return response, response.source_nodes
    except Exception as e:
        logger.error(f"Error querying {source_type}: {e}", exc_info=True)
        st.write(f"Error querying {source_type}: {e}")
        return None, None

#################################
# Streamlit UI for Interaction
#################################

input_text = st.text_input("Ask your question!")

if input_text.strip():
    st.write("Processing the question...")
    st.write(f"Input Received: {input_text}")

    manual_response, manual_sources = query_llama_index(input_text, 'manual')
    log_response, log_sources = query_llama_index(input_text, 'log')

    st.write("### Final Combined Responses ###")

    if manual_response:
        st.write("**From Manuals (PDFs):**", manual_response.response)

        # Print PDF references
        st.write("**PDF Source References:**")
        if manual_sources:
            pdf_files = set()
            for node in manual_sources:
                meta = node.node.metadata
                if meta.get("doc_type") == "manual":
                    pdf_files.add(meta.get("source_file"))
            if pdf_files:
                for f in pdf_files:
                    st.write(f" - Source File: {f}")
            else:
                st.write("No PDF references found.")
        else:
            st.write("No PDF source nodes found.")

    if log_response:
        st.write("**From Maintenance Logs (CSV):**", log_response.response)

        # Print CSV references with details
        st.write("**CSV Source References:**")
        if log_sources and csv_df is not None:
            referenced_rows = set()
            for node in log_sources:
                meta = node.node.metadata
                if meta.get("doc_type") == "log":
                    row_id = meta.get("row_id")
                    if row_id is not None and row_id in csv_df.index:
                        referenced_rows.add(row_id)

            if referenced_rows:
                for rid in referenced_rows:
                    row = csv_df.loc[rid]
                    date = row.get('Date', '')
                    symptom = row.get('Symptom', '')
                    cause = row.get('Possible Cause', '')
                    action = row.get('Corrective Action Taken', '')
                    st.write(f"Row ID: {rid}")
                    st.write(f"- Date: {date}")
                    st.write(f"- Symptom: {symptom}")
                    st.write(f"- Possible Cause: {cause}")
                    st.write(f"- Corrective Action Taken: {action}")
                    st.write("---")
            else:
                st.write("No CSV references found.")
        else:
            st.write("No CSV source nodes found or CSV data not available.")
else:
    st.write("No valid input provided.")
