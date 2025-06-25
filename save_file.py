from fpdf import FPDF
from docx import Document
import time
import os

print("Save file running...")

def get_downloads_folder():
    """Returns the user's Downloads folder path (or the current directory if not found)."""
    downloads_folder = os.path.join(os.environ.get("USERPROFILE", os.getcwd()), "Downloads")
    if not os.path.exists(downloads_folder):
        downloads_folder = os.getcwd()
    return downloads_folder

def save_text(text):
    """
    Saves text in TXT, PDF, or DOCX format.
    The file is saved in the user's Downloads folder.
    """
    print("\nChoose file format to save:")
    print("1. TXT")
    print("2. PDF")
    print("3. DOCX")

    choice = input("Enter 1, 2, or 3: ").strip()

    timestamp = int(time.time())
    base_filename = f"document_{timestamp}"
    downloads_folder = get_downloads_folder()

    if choice == '1':
        filename = os.path.join(downloads_folder, f"{base_filename}.txt")
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Saved as {filename}")

    elif choice == '2':
        filename = os.path.join(downloads_folder, f"{base_filename}.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        
        # Write each line in the text to the PDF using multi_cell
        for line in text.split('\n'):
            pdf.multi_cell(0, 10, line.encode("latin-1", "replace").decode("latin-1"))
        
        pdf.output(filename)
        print(f"Saved as {filename}")

    elif choice == '3':
        filename = os.path.join(downloads_folder, f"{base_filename}.docx")
        doc = Document()
        doc.add_paragraph(text)
        doc.save(filename)
        print(f"Saved as {filename}")

    else:
        print("Invalid choice. Saving as TXT by default.")
        filename = os.path.join(downloads_folder, f"{base_filename}.txt")
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Saved as {filename}")