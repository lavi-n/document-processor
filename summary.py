from transformers import pipeline, AutoTokenizer

print("Summary running...")

# Load summarization pipeline and tokenizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def user_summary_input():
    """Gets user-defined summary length."""
    try:
        max_length = int(input("Max summary length: ")) * 1.33
        min_length = int(input("Min summary length: ")) * 1.33
    except ValueError:
        print("Invalid input. Using default lengths.")
        max_length = 130
        min_length = 30

    return max_length, min_length

def calculate_summary_lengths(text):
    """Adjusts summary length dynamically based on actual token count."""
    tokens = tokenizer.encode(text, truncation=False)
    token_count = len(tokens)

    if token_count < 10: 
        return token_count - 1, max(token_count // 2, 1)

    max_length = min(int(token_count * 0.3), token_count - 1)
    min_length = max(int(max_length * 0.5), 20)

    return max_length, min_length

def choose_summary_length(text):
    """Allows user selection of summary length while ensuring valid limits."""
    print("\nChoose summary length option:")
    print("1. Auto-calculate summary length")
    print("2. Set summary length manually")

    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        max_length, min_length = calculate_summary_lengths(text)
    elif choice == '2':
        max_length, min_length = user_summary_input()
    else:
        print("Invalid choice. Defaulting to auto-calculated lengths.")
        max_length, min_length = calculate_summary_lengths(text)

    return max_length, min_length

def split_text(text, max_chunk_length=900):
    """
    Splits text into manageable chunks by paragraphs.
    Filters out empty lines to avoid very short chunks.
    """
    # Remove leading/trailing whitespace/newlines
    text = text.strip()

    # Split into paragraphs, stripping each and dropping empties.
    paragraphs = [para.strip() for para in text.split('\n') if para.strip()]
    print("Paragraphs:", paragraphs)  # Debug: Verify paragraphs

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_chunk_length:
            current_chunk += para + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out any chunk that is too short (e.g., less than 10 tokens)
    filtered_chunks = []
    for idx, chunk in enumerate(chunks):
        token_count = len(tokenizer.encode(chunk, truncation=False))
        print(f"Chunk {idx} length (tokens): {token_count}")
        if token_count > 10:
            filtered_chunks.append(chunk)
        else:
            print(f"Excluding chunk {idx} as it is too short.")
    return filtered_chunks

def summarize_text(text):
    """Summarizes input text using Hugging Face BART while preventing hallucinations."""
    chunks = split_text(text)
    num_chunks = max(len(chunks), 1)

    total_max_length, total_min_length = choose_summary_length(text)

    if num_chunks == 1:
        chunk_max_length = min(total_max_length, len(text) // 2)
    else:
        chunk_max_length = max(int(total_max_length / num_chunks), 50)
    
    chunk_min_length = max(int(total_min_length / num_chunks), 20)

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=chunk_max_length, min_length=chunk_min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    full_summary = "\n\n".join(summaries)
    return full_summary

def run_summary(text):
    """Main function for text summarization."""
    summary = summarize_text(text)

    if summary:
        return summary
    else:
        return ""