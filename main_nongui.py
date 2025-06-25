from file_extractor import run_file_extractor
from summary import run_summary
from translate_languages import run_translation
from text2speech import save_audio
from save_file import save_text

def main():
    """Runs the main interactive process for working with text."""
    print("Welcome to PurrfectPages!")
    
    file_text = run_file_extractor()
    if not file_text:
        print("\nNo text extracted. Exiting.")
        return

    while True:
        print("\nChoose an option:")
        print("1. Summary")
        print("2. Translate")
        print("3. Text-to-Speech")
        print("4. Save File")
        print("5. Exit")

        choice = input("Enter the number of your option: ").strip()

        if choice == '1':  # Summary
            summarized_text = run_summary(file_text)
            print("\nSummary:\n", summarized_text)
            file_text = summarized_text  # Allow further summarization

        elif choice == '2':  # Translate
            translated_text = run_translation(file_text)
            print("\nTranslated Text:\n", translated_text)
            file_text = translated_text  # Allow further operations on translated output

        elif choice == '3':  # Text-to-Speech
            save_audio(file_text)

        elif choice == '4':  # Save File
            save_text(file_text)
        
        elif choice == '5':  # Exit
            print("\nExiting Document Processor.")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()