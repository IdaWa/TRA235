import fitz  # PyMuPDF for PDF parsing

def extract_text_from_pdf_debug(pdf_path):
    """
    Extracts raw text from the PDF and logs it for debugging purposes.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if page_text:
            print(f"--- Page {page_num} ---")
            print(page_text)  # Print raw text from each page for debugging
            text += f"--- Page {page_num} ---\n{page_text}\n"
        else:
            print(f"--- Page {page_num} has no text or could not be extracted ---")
    return text

# Define the PDF path
pdf_path = r'C:\Users\ida\OneDrive\Skrivbord\TRA235\Docs_pdfs\Bar Feeder - Maintenance - Haas Service Manual.pdf'

# Extract the text
raw_text = extract_text_from_pdf_debug(pdf_path)

# Save the raw extracted text to a file for further analysis
with open("extracted_text_debug.txt", "w", encoding="utf-8") as debug_file:
    debug_file.write(raw_text)

print("Raw text extraction completed. Check 'extracted_text_debug.txt' for details.")
