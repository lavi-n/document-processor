import langid
from transformers import MarianMTModel, MarianTokenizer

print("Language translator running")

# Set langid to focus only on English detection
langid.set_languages(['en'])

# Available language pairs
language_pairs = {
    "English → Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "English → Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English → French": "Helsinki-NLP/opus-mt-en-fr",
    "English → German": "Helsinki-NLP/opus-mt-en-de",
    "English → Marathi": "Helsinki-NLP/opus-mt-en-mr",
    "English → Tamil": "Helsinki-NLP/opus-mt-en-ta",
    "English → Bengali": "Helsinki-NLP/opus-mt-en-bn",
    "English → Gujarati": "Helsinki-NLP/opus-mt-en-gu",
    "English → Malayalam": "Helsinki-NLP/opus-mt-en-ml",
    "English → Telugu": "Helsinki-NLP/opus-mt-en-te",
    "English → Kannada": "Helsinki-NLP/opus-mt-en-kn",
    "English → Punjabi": "Helsinki-NLP/opus-mt-en-pa",
    "English → Italian": "Helsinki-NLP/opus-mt-en-it",
    "English → Dutch": "Helsinki-NLP/opus-mt-en-nl",
    "English → Portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "English → Russian": "Helsinki-NLP/opus-mt-en-ru",
    "English → Arabic": "Helsinki-NLP/opus-mt-en-ar",
    "English → Chinese": "Helsinki-NLP/opus-mt-en-zh",
    "English → Japanese": "Helsinki-NLP/opus-mt-en-ja"
}

def detect_language(text):
    """Detects if the input language is English; otherwise, returns 'unknown'."""
    detected_lang = langid.classify(text)[0]
    return detected_lang if detected_lang == "en" else "unknown"

def translate_text(text, language_choice):
    """Translates text only if it's in English; otherwise, returns the original text."""
    if detect_language(text) != "en":
        print("\nThe input text is not in English. Returning original text.")
        return text

    model_name = language_pairs.get(language_choice)

    # Debugging output to verify model selection
    print(f"\nSelected Language: {language_choice}")
    print(f"Model Name: {model_name}")

    if not model_name:
        print("\nUnsupported language pair. Returning original text.")
        return text

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        translated = model.generate(**inputs)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"\nError during translation: {e}. Returning original text.")
        return text

def run_translation(text):
    """Handles translation while ensuring input is English."""
    print("\nAvailable Translation Options:")
    for i, lang in enumerate(language_pairs.keys(), 1):
        print(f"{i}. {lang}")

    choice_num = input("\nEnter the number of your chosen translation option: ").strip()

    try:
        choice_num = int(choice_num)
        language_choice = list(language_pairs.keys())[choice_num - 1]
        return translate_text(text, language_choice)
    except (ValueError, IndexError):
        print("\nInvalid choice. Returning original text.")
        return text