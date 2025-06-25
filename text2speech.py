import pyttsx3
import time
import wave
import os
import shutil

print("Text-to-Speech running...")

# Global engine instance for reuse
engine = None

def configure_audio_engine():
    """
    Configures the pyttsx3 engine once using the user's customizations, then reuses it.
    """
    global engine

    if engine is not None:
        return engine  # Return already-configured engine

    engine = pyttsx3.init()

    # Predefined voices and their display names
    voice_options = [
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0",
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-GB_HAZEL_11.0"
    ]
    voice_names = [
        "Microsoft David (English - US)",
        "Microsoft Hazel (English - GB)",
        "Microsoft Zira (English - US)"
    ]

    print("\nAvailable Voices:")
    for i, name in enumerate(voice_names):
        print(f"{i+1}: {name}")

    try:
        choice = int(input("\nEnter the number of your chosen voice: ").strip())
        if choice - 1 not in range(len(voice_options)):
            raise ValueError
        selected_voice = voice_options[choice - 1]
    except ValueError:
        print("\nInvalid choice. Using system default (Microsoft David).")
        selected_voice = voice_options[0]

    engine.setProperty('voice', selected_voice)

    try:
        rate = int(input("Enter speech speed (default is 150, range 100-200): "))
    except ValueError:
        print("Invalid input. Using default speed of 150.")
        rate = 150
    engine.setProperty('rate', rate)

    try:
        volume = float(input("Enter volume (0.0 to 1.0, default is 1.0): "))
        if not 0.0 <= volume <= 1.0:
            raise ValueError
    except ValueError:
        print("Invalid input. Using default volume of 1.0.")
        volume = 1.0
    engine.setProperty('volume', volume)

    print("\nAudio engine configured with selected voice.\n")
    return engine

def get_wav_duration(wav_file):
    """Returns the duration (in seconds) of a WAV file."""
    try:
        with wave.open(wav_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except wave.Error as e:
        print(f"Error reading WAV file: {e}")
        return 0

def save_audio(text):
    """
    Saves the synthesized audio file directly to the user's Downloads folder.
    - If duration > 0 seconds, saves the file, plays it, and shows "Your download is ready."
    - If duration is 0 seconds, gracefully informs the user.
    """
    global engine
    if engine is None:
        engine = configure_audio_engine()

    timestamp = int(time.time())

    # Determine the Downloads folder.
    downloads_folder = os.path.join(os.environ.get("USERPROFILE", os.getcwd()), "Downloads")
    if not os.path.exists(downloads_folder):
        downloads_folder = os.getcwd()  # Fallback if Downloads folder is not found

    wav_file = os.path.join(downloads_folder, f"audio_{timestamp}.wav")

    # Synthesize audio and save it directly to the Downloads folder.
    engine.save_to_file(text, wav_file)
    engine.runAndWait()  # Wait for the saving process to complete

    # Check the duration of the saved audio.
    duration = get_wav_duration(wav_file)
    if duration > 0:
        print(f"\nYour download is ready")

        # Play the saved audio using the same settings
        engine.say(text)
        engine.runAndWait()
    else:
        print("We currently support only English at this time, but stay tuned for updates!")