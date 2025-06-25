LuminaText is a document processing application designed to help you interact with text in various ways. It offers functionalities for text extraction, summarization, translation, text-to-speech conversion, and saving processed text into different file formats. The application provides both a graphical user interface (GUI) and a command-line interface (CLI) for flexibility.

## Features

* **Text Extraction**: Extract text from a variety of file formats, including:
    * PDF documents
    * Image files (PNG, JPG, JPEG) using OCR
    * Plain Text (TXT) files
    * Microsoft Word documents (DOCX)
* **Text Summarization**: Generate concise summaries of lengthy texts using a Hugging Face BART model. Users can choose between auto-calculated or manually set summary lengths.
* **Text Translation**: Translate extracted or processed text into different languages.
* **Text-to-Speech (TTS)**: Convert any text into spoken audio and save it as a WAV file. Users can configure voice, speed, and volume.
* **File Saving**: Save your processed text into the following formats:
    * Plain Text (TXT)
    * PDF documents
    * Microsoft Word documents (DOCX)

## Installation

To run LuminaText, you need to install the required dependencies.

1.  **Install dependencies:**
    The project uses the following Python libraries:
    * `PySide6`
    * `pdfplumber`
    * `easyocr`
    * `python-docx`
    * `pyttsx3`
    * `fpdf`
    * `transformers`
    * `langid`
    * `torch`

    You can install them using pip. Navigate to the directory containing `requirements.txt` and run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

LuminaText can be used via its GUI or through a command-line interface.

### Graphical User Interface (GUI)

Run the `main.py` file to launch the GUI application:
```bash
python main.py
```
The GUI provides an interactive experience for all features.

### Command-Line Interface (CLI)

Run the `main_nongui.py` file to use the CLI version:
```bash
python main_nongui.py
```
The CLI will guide you through the available options: text extraction, summarization, translation, text-to-speech, and saving files.
