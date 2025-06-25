import pdfplumber
import easyocr
import os
import docx

print("File extractor started...")

def extract_text(file_path):
    """
    Determines file type and calls relevant functions
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("File does not exist.")

    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, image, TXT, or DOCX.")

def extract_text_from_docx(file_path):
    """
    Extract text from a docx file using docx.
    """
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF using pdfplumber.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text is not None and page_text.strip():
                    text += page_text + "\n"
        if not text.strip():
            raise ValueError("No readable text found in PDF.")
        return text
    except Exception as e:
        raise Exception(f"Error processing PDF: {e}")

def extract_text_from_image(image_path, languages=['en']):
    """
    Extract text from an image using easyocr.
    """
    reader = easyocr.Reader(['en', 'hi'])
    result = reader.readtext(image_path, detail=0)  # detail=0 only text
    extracted_text = " ".join(result)
    return extracted_text

def run_file_extractor():
    """
    Prompts user for file path and handles extraction.
    """
    file_path = input("Enter the file path to extract text: ").strip()
    try:
        text = extract_text(file_path)
        print("\nExtracted Text:\n", text)
        return text
    except Exception as e:
        print(f"\nError: {e}")
        return None