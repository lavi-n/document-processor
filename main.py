import sys
import os
import time
import wave
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QFileDialog, QLabel,
    QComboBox, QMessageBox, QDialog, QFormLayout, QSpacerItem, QSizePolicy 
)
from PySide6.QtCore import QObject, Signal, Qt, QRunnable, QThreadPool
from PySide6.QtGui import QIntValidator, QDoubleValidator, QFont 

from file_extractor import extract_text_from_pdf, extract_text_from_image, extract_text_from_docx
# summary and translate_languages imported later for lazy loading
import pyttsx3 
from fpdf import FPDF 
from docx import Document 

tts_engine = None 
current_tts_worker = None 

# Light theme
QSS_LIGHT_MODE = """
QMainWindow {
    background-color: #F0F8FF;
}

QLabel {
    color: #4682B4; 
    font-family: 'Georgia', 'Times New Roman', serif;
}

QPushButton {
    background-color: #ADD8E6; 
    color: #000080; 
    border-radius: 12px; 
    padding: 8px 18px; 
    font-weight: 600; 
    border: none; 
}

QPushButton:hover {
    background-color: #87CEEB; 
}

QPushButton:pressed {
    background-color: #A2D9CE; 
}

QTextEdit, QLineEdit, QComboBox {
    border: 1px solid #B0E0E6; 
    border-radius: 8px;
    padding: 4px;
    background-color: #FFFFFF; 
    color: #191970; 
}

QComboBox::drop-down {
    border: 0px; 
}

QComboBox::down-arrow {
    image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAYAAADg/NUuAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8AgAAADdJREFUGFdj/M/AATFgwAAeGhqaiwGA+D/y+n+M/B9A+H8i5v//MvC/Q+P/MGAALh0ZGBggAMhPHz504wUAAP7bB89gN9kAAAAASUVORK5CYII=);
    width: 12px;
    height: 12px;
}

QStatusBar {
    background-color: #E0FFFF; 
    color: #00008B; 
    font-size: 10pt;
}

#WelcomeLabel {
    color: #404040; 
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 30pt; 
    font-weight: 500;
}

#StartButton {
    background-color: #77a1c4; 
    color: #FFFFFF; 
    border-radius: 25px; 
    padding: 10px 25px; 
    font-weight: 500;
    font-size: 20pt; 
    border: 3px solid #4d687d; 
}

#StartButton:hover {
    background-color: #6488a4; 
    color: #FFFFFF; 
    border: 3px solid #4d687d;
}

#StartButton:pressed {
    background-color: #57768e;
    border: 3px solid #4d687d;
}
"""

# Dark theme
QSS_DARK_MODE = """
QMainWindow {
    background-color: #2C3E50;
    color: #ECF0F1; 
}

QLabel {
    color: #ECF0F1; 
    font-family: 'Georgia', 'Times New Roman', serif;
}

QPushButton {
    background-color: #34495E; 
    color: #BDC3C7; 
    border-radius: 12px;
    padding: 8px 18px;
    font-weight: 600;
    border: none;
}

QPushButton:hover {
    background-color: #4A6572; 
}

QPushButton:pressed {
    background-color: #5D7987; 
}

QTextEdit, QLineEdit, QComboBox {
    border: 1px solid #7F8C8D; 
    border-radius: 8px;
    padding: 4px;
    background-color: #1A2632; 
    color: #ECF0F1; 
}

QComboBox::drop-down {
    border: 0px;
}

QComboBox::down-arrow {
    image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAYAAADg/NUuAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8AgAAADdJREFUGFdj/M/AATFgwAAeGhqaiwGA+D/y+n+M/B9A+H8i5v//MvC/Q+P/MGAALh0ZGBggAMhPHz504wUAAP7bB89gN9kAAAAASUVORK5CYII=); 
    width: 12px;
    height: 12px;
    filter: invert(100%); 
}

QStatusBar {
    background-color: #34495E; 
    color: #ECF0F1; 
    font-size: 10pt;
}
"""

# QMessageBox buttons
QSS_MESSAGE_BOX_BUTTONS_LIGHT = """
QPushButton {
    background-color: #ADD8E6; 
    color: #000080;
    border-radius: 12px;
    padding: 8px 18px;
    font-weight: 600;
    border: none;
}
QPushButton:hover {
    background-color: #87CEEB; 
}
QPushButton:pressed {
    background-color: #A2D9CE; 
}
"""

QSS_MESSAGE_BOX_BUTTONS_DARK = """
QPushButton {
    background-color: #34495E; 
    color: #BDC3C7; 
    border-radius: 12px;
    padding: 8px 18px;
    font-weight: 600;
    border: none;
}
QPushButton:hover {
    background-color: #4A6572; 
}
QPushButton:pressed {
    background-color: #5D7987; 
}
"""


def get_downloads_folder_path():
    """Returns the user's Downloads folder path (or the current directory if not found)."""
    downloads_folder = os.path.join(os.environ.get("USERPROFILE", os.getcwd()), "Downloads")
    if not os.path.exists(downloads_folder):
        downloads_folder = os.getcwd()
    return downloads_folder

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker QRunnable.
    """
    text_extracted = Signal(str)
    summary_completed = Signal(str)
    translation_completed = Signal(str)
    tts_completed = Signal(str)
    file_saved = Signal(str)
    error_occurred = Signal(str)
    loading_message = Signal(str)
    stop_tts_request = Signal()


class FileExtractionWorker(QRunnable):
    """Worker to extract text from a file in a separate thread."""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.signals = WorkerSignals()

    def run(self):
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError("File does not exist.")

            text = ""
            self.signals.loading_message.emit("Extracting text from file...")
            if self.file_path.endswith(".pdf"):
                text = extract_text_from_pdf(self.file_path)
            elif self.file_path.endswith((".png", ".jpg", ".jpeg")):
                text = extract_text_from_image(self.file_path)
            elif self.file_path.endswith(".txt"):
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            elif self.file_path.endswith(".docx"):
                text = extract_text_from_docx(self.file_path)
            else:
                raise ValueError("Unsupported file type. Please upload a PDF, image, TXT, or DOCX.")

            self.signals.text_extracted.emit(text)
        except Exception as e:
            self.signals.error_occurred.emit(f"Error extracting text: {e}")

class SummarizationWorker(QRunnable):
    """Worker to summarize text in a separate thread."""
    _summarizer_pipeline = None 
    _tokenizer = None 

    def __init__(self, text, max_len=None, min_len=None):
        super().__init__()
        self.text = text
        self.max_len = max_len
        self.min_len = min_len
        self.signals = WorkerSignals()

    def run(self):
        try:
            if SummarizationWorker._summarizer_pipeline is None:
                from transformers import pipeline, AutoTokenizer
                self.signals.loading_message.emit("Loading summarization model (first time may take a moment)...")
                SummarizationWorker._summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
                SummarizationWorker._tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
                self.signals.loading_message.emit("Summarization model loaded.")

            tokens = SummarizationWorker._tokenizer.encode(self.text, truncation=False)
            token_count = len(tokens)

            if self.max_len is None or self.min_len is None:
                if token_count < 10:
                    self.max_len = token_count - 1
                    self.min_len = max(token_count // 2, 1)
                else:
                    self.max_len = min(int(token_count * 0.3), token_count - 1)
                    self.min_len = max(int(self.max_len * 0.5), 20)

            if self.min_len > self.max_len:
                self.min_len = self.max_len // 2

            from summary import split_text
            chunks = split_text(self.text, max_chunk_length=900)
            num_chunks = max(len(chunks), 1)

            if num_chunks == 1:
                chunk_max_length = min(self.max_len, len(self.text) // 2) if self.max_len else len(self.text) // 2
            else:
                chunk_max_length = max(int(self.max_len / num_chunks), 50) if self.max_len else 50
            
            chunk_min_length = max(int(self.min_len / num_chunks), 20) if self.min_len else 20

            summaries = []
            self.signals.loading_message.emit("Generating summary...")
            for chunk in chunks:
                summary_output = SummarizationWorker._summarizer_pipeline(
                    chunk,
                    max_length=int(chunk_max_length),
                    min_length=int(chunk_min_length),
                    do_sample=False
                )
                summaries.append(summary_output[0]['summary_text'])

            full_summary = "\n\n".join(summaries)
            self.signals.summary_completed.emit(full_summary)
        except Exception as e:
            self.signals.error_occurred.emit(f"Error summarizing text: {e}")

class TranslationWorker(QRunnable):
    """Worker to translate text in a separate thread."""
    _translation_model = None
    _translation_tokenizer = None
    _langid_initialized = False

    def __init__(self, text, target_language_key):
        super().__init__()
        self.text = text
        self.target_language_key = target_language_key
        self.signals = WorkerSignals()

    def run(self):
        try:
            from translate_languages import detect_language, language_pairs, MarianMTModel, MarianTokenizer, langid

            if not TranslationWorker._langid_initialized:
                langid.set_languages(['en'])
                TranslationWorker._langid_initialized = True

            detected_lang = detect_language(self.text)
            if detected_lang != "en":
                self.signals.error_occurred.emit("The input text is not detected as English. Translation might not work correctly. Please provide English text for translation.")
                return

            model_name = language_pairs.get(self.target_language_key)
            if not model_name:
                self.signals.error_occurred.emit("Unsupported language pair. Cannot translate.")
                return

            if TranslationWorker._translation_model is None or \
               (TranslationWorker._translation_model and TranslationWorker._translation_model.name != model_name):
                self.signals.loading_message.emit(f"Loading translation model for {self.target_language_key} (first time may take a moment)...")
                TranslationWorker._translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
                TranslationWorker._translation_model = MarianMTModel.from_pretrained(model_name)
                TranslationWorker._translation_model.name = model_name
                self.signals.loading_message.emit(f"Translation model for {self.target_language_key} loaded.")

            self.signals.loading_message.emit(f"Translating text to {self.target_language_key}...")
            inputs = TranslationWorker._translation_tokenizer(self.text, return_tensors="pt", truncation=True, padding=True)
            translated = TranslationWorker._translation_model.generate(**inputs)
            translated_text = TranslationWorker._translation_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            self.signals.translation_completed.emit(translated_text)
        except Exception as e:
            self.signals.error_occurred.emit(f"Error translating text: {e}")

class TextToSpeechWorker(QRunnable):
    """Worker to convert text to speech and save/play in a separate thread."""
    def __init__(self, text, voice_id, rate, volume):
        super().__init__()
        self.text = text
        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume
        self.signals = WorkerSignals()
        self._is_stopped = False # Check if stopping requested

        self.signals.stop_tts_request.connect(self.stop_playback)

    def stop_playback(self):
        """Method to be called when stop_tts_request signal is emitted."""
        global tts_engine
        if tts_engine:
            tts_engine.stop()
            self._is_stopped = True 

    def run(self):
        global tts_engine
        global current_tts_worker
        current_tts_worker = self 

        try:
            # Re-initializes the engine to ensure clean playback.
            if tts_engine:
                try:
                    tts_engine.endLoop() 
                except RuntimeError:
                    pass 
                tts_engine = None 

            tts_engine = pyttsx3.init() 

            voices = tts_engine.getProperty('voices')
            selected_voice_obj = None
            for voice in voices:
                if voice.id == self.voice_id:
                    selected_voice_obj = voice
                    break
            if selected_voice_obj:
                tts_engine.setProperty('voice', selected_voice_obj.id)
            else:
                self.signals.error_occurred.emit("Selected voice not found. Using default voice.")
                tts_engine.setProperty('voice', voices[0].id if voices else '')

            tts_engine.setProperty('rate', self.rate)
            tts_engine.setProperty('volume', self.volume)

            timestamp = int(time.time())
            downloads_folder = get_downloads_folder_path()
            wav_file = os.path.join(downloads_folder, f"audio_{timestamp}.wav")

            # Generate audio file
            self.signals.loading_message.emit("Generating audio file...")
            tts_engine.save_to_file(self.text, wav_file)
            tts_engine.runAndWait() 

            # Audio saved message
            self.signals.loading_message.emit(f"Audio saved to: {wav_file}")

            if self._is_stopped: # If stopped during generation phase
                self.signals.tts_completed.emit("Audio playback stopped by user.")
                return

            # Play audio
            self.signals.loading_message.emit(f"Playing audio...")
            
            duration = self.get_wav_duration(wav_file)
            if duration > 0:
                tts_engine.say(self.text)
                tts_engine.runAndWait() 
                if self._is_stopped: 
                    self.signals.tts_completed.emit("Audio playback stopped by user.")
                    return
                else: 
                    self.signals.tts_completed.emit("Audio playback completed.")
            else:
                self.signals.error_occurred.emit("Failed to generate audio or audio duration is zero. Ensure text is not empty and voice is supported.")
        except Exception as e:
            self.signals.error_occurred.emit(f"Error during Text-to-Speech: {e}")
        finally:
            if current_tts_worker == self:
                current_tts_worker = None


    def get_wav_duration(self, wav_file):
        """Returns the duration (in seconds) of a WAV file."""
        try:
            with wave.open(wav_file, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                return duration
        except wave.Error as e:
            return 0

class SaveFileWorker(QRunnable):
    """Worker to save text to a file in a separate thread."""
    def __init__(self, text, file_format):
        super().__init__()
        self.text = text
        self.file_format = file_format
        self.signals = WorkerSignals()

    def run(self):
        try:
            timestamp = int(time.time())
            base_filename = f"document_{timestamp}"
            downloads_folder = get_downloads_folder_path()
            filename = ""

            self.signals.loading_message.emit(f"Saving file as {self.file_format}...")
            if self.file_format == 'TXT':
                filename = os.path.join(downloads_folder, f"{base_filename}.txt")
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(self.text)
            elif self.file_format == 'PDF':
                filename = os.path.join(downloads_folder, f"{base_filename}.pdf")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", size=12)
                for line in self.text.split('\n'):
                    try:
                        pdf.multi_cell(0, 10, line)
                    except UnicodeEncodeError:
                        pdf.multi_cell(0, 10, line.encode("latin-1", "replace").decode("latin-1"))
                pdf.output(filename)
            elif self.file_format == 'DOCX':
                filename = os.path.join(downloads_folder, f"{base_filename}.docx")
                doc = Document()
                doc.add_paragraph(self.text)
                doc.save(filename)
            else:
                raise ValueError("Unsupported file format.")

            self.signals.file_saved.emit(f"File saved as {filename}")
        except Exception as e:
            self.signals.error_occurred.emit(f"Error saving file: {e}")

# User Input
class SummaryLengthDialog(QDialog):
    def __init__(self, parent=None, token_count=0):
        super().__init__(parent)
        self.setWindowTitle("Summary Length Options")
        self.setFixedSize(300, 180)

        layout = QVBoxLayout(self)
        self.auto_radio = QPushButton("Auto-calculate length")
        self.manual_radio = QPushButton("Set length manually")
        self.auto_radio.setCheckable(True)
        self.manual_radio.setCheckable(True)
        self.auto_radio.setChecked(True)

        self.max_length_input = QLineEdit()
        self.min_length_input = QLineEdit()
        self.max_length_input.setPlaceholderText("Max length (words)")
        self.min_length_input.setPlaceholderText("Min length (words)")
        self.max_length_input.setValidator(QIntValidator())
        self.min_length_input.setValidator(QIntValidator())

        self.manual_inputs_layout = QFormLayout()
        self.manual_inputs_layout.addRow("Max Length:", self.max_length_input)
        self.manual_inputs_layout.addRow("Min Length:", self.min_length_input)

        self.manual_inputs_widget = QWidget()
        self.manual_inputs_widget.setLayout(self.manual_inputs_layout)
        self.manual_inputs_widget.setVisible(False)

        self.auto_radio.clicked.connect(lambda: self.manual_inputs_widget.setVisible(False))
        self.manual_radio.clicked.connect(lambda: self.manual_inputs_widget.setVisible(True))

        self.token_count_label = QLabel(f"Detected text token count: {token_count}")
        layout.addWidget(self.token_count_label)
        layout.addWidget(self.auto_radio)
        layout.addWidget(self.manual_radio)
        layout.addWidget(self.manual_inputs_widget)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.max_len = None
        self.min_len = None

    def get_summary_lengths(self):
        if self.auto_radio.isChecked():
            return None, None
        else:
            try:
                max_len = int(self.max_length_input.text()) * 1.33 if self.max_length_input.text() else None
                min_len = int(self.min_length_input.text()) * 1.33 if self.min_length_input.text() else None
                return max_len, min_len
            except ValueError:
                return None, None 

class TTSConfigurationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text-to-Speech Settings")
        self.setFixedSize(400, 250)

        layout = QVBoxLayout(self)

        voice_label = QLabel("Voice:")
        self.voice_combo = QComboBox()
        self.voice_map = {}
        try:
            engine_temp = pyttsx3.init()
            voices = engine_temp.getProperty('voices')
            for i, voice in enumerate(voices):
                display_name = f"{voice.name} ({voice.id})"
                self.voice_combo.addItem(display_name, voice.id)
                self.voice_map[voice.id] = voice.name
            engine_temp.stop()
        except Exception as e:
            QMessageBox.warning(self, "TTS Error", f"Could not load TTS voices: {e}\nFalling back to default.")
            self.voice_combo.addItem("Default Voice", "default")
            self.voice_map["default"] = "Default Voice"

        rate_label = QLabel("Speech Speed (100-200):")
        self.rate_input = QLineEdit("150")
        self.rate_input.setValidator(QIntValidator(100, 200))

        volume_label = QLabel("Volume (0.0-1.0):")
        self.volume_input = QLineEdit("1.0")
        self.volume_input.setValidator(QDoubleValidator(0.0, 1.0, 2))

        form_layout = QFormLayout()
        form_layout.addRow(voice_label, self.voice_combo)
        form_layout.addRow(rate_label, self.rate_input)
        form_layout.addRow(volume_label, self.volume_input)
        layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def get_tts_settings(self):
        voice_id = self.voice_combo.currentData()
        rate = int(self.rate_input.text()) if self.rate_input.text() else 150
        volume = float(self.volume_input.text()) if self.volume_input.text() else 1.0
        return voice_id, rate, volume

# Welcome Window
class WelcomeWindow(QMainWindow):
    start_app_signal = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to LuminaText!")
        self.setGeometry(300, 200, 600, 400) 

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignCenter) 

        welcome_label = QLabel("Welcome to LuminaText!")
        welcome_label.setObjectName("WelcomeLabel") 
        welcome_label.setAlignment(Qt.AlignCenter)
        
        main_layout.addWidget(welcome_label)

        start_button = QPushButton("Start")
        start_button.setObjectName("StartButton") 
        start_button.setFixedSize(150, 50) 
        
        start_button.clicked.connect(self.start_app)
        main_layout.addWidget(start_button, alignment=Qt.AlignCenter)

    def start_app(self):
        self.start_app_signal.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LuminaText Document Processor")
        self.setGeometry(100, 100, 1000, 700)

        self.current_text = ""
        self.is_dark_mode = False # Current theme

        self.ml_thread_pool = QThreadPool()
        self.ml_thread_pool.setMaxThreadCount(1) # Summarization and Translation

        self.tts_thread_pool = QThreadPool()
        self.tts_thread_pool.setMaxThreadCount(1) # TTS

        self.save_thread_pool = QThreadPool()
        self.save_thread_pool.setMaxThreadCount(1) # Saving files

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_bar_layout = QHBoxLayout()
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("No file selected")
        self.file_path_input.setReadOnly(True)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        top_bar_layout.addWidget(QLabel("File Path:"))
        top_bar_layout.addWidget(self.file_path_input)
        top_bar_layout.addWidget(browse_button)
        
        top_bar_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.dark_mode_button = QPushButton("Dark Mode")
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)
        self.dark_mode_button.setFixedSize(120, 40) 
        top_bar_layout.addWidget(self.dark_mode_button)

        main_layout.addLayout(top_bar_layout)

        self.stop_audio_button = QPushButton("Stop Audio")
        self.stop_audio_button.clicked.connect(self.stop_current_tts_playback)
        self.stop_audio_button.setEnabled(False)
        self.stop_audio_button.hide() 
        main_layout.addWidget(self.stop_audio_button)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(False)
        self.text_display.setPlaceholderText("Extracted text and results will appear here. You can also type or paste text here.")
        main_layout.addWidget(self.text_display)

        actions_layout = QHBoxLayout()

        summary_button = QPushButton("Summarize")
        summary_button.clicked.connect(self.summarize_document)
        actions_layout.addWidget(summary_button)

        translate_layout = QVBoxLayout()
        self.translate_combo = QComboBox()
        from translate_languages import language_pairs
        for lang_pair in language_pairs.keys():
            self.translate_combo.addItem(lang_pair)
        translate_button = QPushButton("Translate")
        translate_button.clicked.connect(self.translate_document)
        translate_layout.addWidget(self.translate_combo)
        translate_layout.addWidget(translate_button)
        actions_layout.addLayout(translate_layout)

        tts_button = QPushButton("Text-to-Speech")
        tts_button.clicked.connect(self.text_to_speech)
        actions_layout.addWidget(tts_button)

        save_layout = QVBoxLayout()
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["TXT", "PDF", "DOCX"])
        save_button = QPushButton("Save File")
        save_button.clicked.connect(self.save_document)
        save_layout.addWidget(self.save_format_combo)
        save_layout.addWidget(save_button)
        actions_layout.addLayout(save_layout)

        main_layout.addLayout(actions_layout)

        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

    def update_status_message(self, message):
        """Updates the status label."""
        self.status_label.setText(message)

    def clear_status_message(self):
        """Clears the status label."""
        self.status_label.setText("Ready")

    def handle_error(self, message):
        global tts_engine
        self.clear_status_message()
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText("Error")
        msg_box.setInformativeText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        
        if self.is_dark_mode:
            msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_DARK)
        else:
            msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_LIGHT)
        
        msg_box.exec()

        if tts_engine:
            try:
                tts_engine.endLoop()
            except RuntimeError:
                pass
            tts_engine = None # Force re-initialization
        self.stop_audio_button.setEnabled(False)
        self.stop_audio_button.hide()


    def update_text_display(self, text):
        self.clear_status_message()
        self.text_display.setText(text)
        self.current_text = text 
        pass 


    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document",
            "",
            "Documents (*.pdf *.docx *.txt *.png *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.file_path_input.setText(file_path)
            self.update_status_message("Extracting text...")
            worker = FileExtractionWorker(file_path)
            worker.signals.text_extracted.connect(self.update_text_display)
            worker.signals.error_occurred.connect(self.handle_error)
            worker.signals.loading_message.connect(self.update_status_message)
            self.ml_thread_pool.start(worker) 

    def summarize_document(self):
        self.current_text = self.text_display.toPlainText()
        if not self.current_text:
            QMessageBox.warning(self, "No Text", "Please extract text from a file, or type/paste text into the text area.")
            return

        from transformers import AutoTokenizer
        temp_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        token_count = len(temp_tokenizer.encode(self.current_text, truncation=False))

        dialog = SummaryLengthDialog(self, token_count)
        dialog.setStyleSheet(QSS_LIGHT_MODE if not self.is_dark_mode else QSS_DARK_MODE)
        if dialog.exec() == QDialog.Accepted:
            max_len, min_len = dialog.get_summary_lengths()
            self.update_status_message("Summarizing text...")
            worker = SummarizationWorker(self.current_text, max_len, min_len)
            worker.signals.summary_completed.connect(self.update_text_display)
            worker.signals.error_occurred.connect(self.handle_error)
            worker.signals.loading_message.connect(self.update_status_message)
            self.ml_thread_pool.start(worker) 

    def translate_document(self):
        self.current_text = self.text_display.toPlainText()
        if not self.current_text:
            QMessageBox.warning(self, "No Text", "Please extract text from a file, or type/paste text into the text area.")
            return

        from translate_languages import detect_language
        detected_lang = detect_language(self.current_text)
        if detected_lang != "en":
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("Translation Warning")
            msg_box.setInformativeText("The input text is not detected as English. Translation might not work correctly. Please provide English text for translation.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            if self.is_dark_mode:
                msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_DARK)
            else:
                msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_LIGHT)
            msg_box.exec()
            return

        target_language_key = self.translate_combo.currentText()
        self.update_status_message(f"Translating to {target_language_key}...")
        worker = TranslationWorker(self.current_text, target_language_key)
        worker.signals.translation_completed.connect(self.update_text_display)
        worker.signals.error_occurred.connect(self.handle_error)
        worker.signals.loading_message.connect(self.update_status_message)
        self.ml_thread_pool.start(worker) # Use ML thread pool

    def text_to_speech(self):
        global current_tts_worker
        self.current_text = self.text_display.toPlainText()
        if not self.current_text:
            QMessageBox.warning(self, "No Text", "Please extract text from a file, or type/paste text into the text area.")
            return

        if self.tts_thread_pool.activeThreadCount() > 0:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText("Audio in Progress")
            msg_box.setInformativeText("Audio playback is already in progress. Please wait or stop the current playback.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            if self.is_dark_mode:
                msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_DARK)
            else:
                msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_LIGHT)
            msg_box.exec()
            return

        dialog = TTSConfigurationDialog(self)
        dialog.setStyleSheet(QSS_LIGHT_MODE if not self.is_dark_mode else QSS_DARK_MODE)
        if dialog.exec() == QDialog.Accepted:
            voice_id, rate, volume = dialog.get_tts_settings()
            self.update_status_message("Preparing audio...")
            self.stop_audio_button.show() 
            self.stop_audio_button.setEnabled(True) 
            worker = TextToSpeechWorker(self.current_text, voice_id, rate, volume)
            current_tts_worker = worker 
            worker.signals.tts_completed.connect(self.handle_tts_completion)
            worker.signals.error_occurred.connect(self.handle_error)
            worker.signals.loading_message.connect(self.update_status_message)
            self.tts_thread_pool.start(worker) 

    def stop_current_tts_playback(self):
        global current_tts_worker
        if current_tts_worker:
            current_tts_worker.signals.stop_tts_request.emit()
            current_tts_worker = None 
            self.update_status_message("Stopping audio playback...")
            self.stop_audio_button.setEnabled(False) 
            self.stop_audio_button.hide() 

    def handle_tts_completion(self, message):
        self.clear_status_message()
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Text-to-Speech")
        msg_box.setInformativeText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        if self.is_dark_mode:
            msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_DARK)
        else:
            msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_LIGHT)
        msg_box.exec()
        self.stop_audio_button.setEnabled(False) 
        self.stop_audio_button.hide() 

    def save_document(self):
        self.current_text = self.text_display.toPlainText()
        if not self.current_text:
            QMessageBox.warning(self, "No Text", "Please extract text from a file, or type/paste text into the text area.")
            return

        file_format = self.save_format_combo.currentText()
        self.update_status_message(f"Saving file as {file_format}...")
        worker = SaveFileWorker(self.current_text, file_format)
        worker.signals.file_saved.connect(self.handle_file_saved)
        worker.signals.error_occurred.connect(self.handle_error)
        worker.signals.loading_message.connect(self.update_status_message)
        self.save_thread_pool.start(worker) 

    def handle_file_saved(self, message):
        self.clear_status_message()
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Save File")
        msg_box.setInformativeText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        if self.is_dark_mode:
            msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_DARK)
        else:
            msg_box.setStyleSheet(QSS_MESSAGE_BOX_BUTTONS_LIGHT)
        msg_box.exec()

    def toggle_dark_mode(self):
        """Toggles between light and dark mode stylesheets."""
        self.is_dark_mode = not self.is_dark_mode
        if self.is_dark_mode:
            QApplication.instance().setStyleSheet(QSS_DARK_MODE)
            self.dark_mode_button.setText("Light Mode")
            self.dark_mode_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #5D7987;  
                    color: #ECF0F1; 
                    border-radius: 12px;
                    padding: 8px 18px;
                    font-weight: 600;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #618491; 
                }
                QPushButton:pressed {
                    background-color: #2E4057; 
                }
                """
            )
        else:
            QApplication.instance().setStyleSheet(QSS_LIGHT_MODE)
            self.dark_mode_button.setText("Dark Mode")
            self.dark_mode_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #ADD8E6; 
                    color: #000080;
                    border-radius: 12px;
                    padding: 8px 18px;
                    font-weight: 600;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #87CEEB;
                }
                QPushButton:pressed {
                    background-color: #A2D9CE;
                }
                """
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    app.setStyleSheet(QSS_LIGHT_MODE)
    
    default_font = QFont("Georgia", 10) 
    app.setFont(default_font)

    welcome_window = WelcomeWindow()
    main_window = MainWindow()

    welcome_window.start_app_signal.connect(welcome_window.hide)
    welcome_window.start_app_signal.connect(main_window.show)

    welcome_window.show()

    sys.exit(app.exec())