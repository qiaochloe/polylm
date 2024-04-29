import re
from data import Vocabulary


def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    paragraphs = re.split(r'\n\s*\n', content)

    # Remove the first line from each paragraph
    processed_paragraphs = []
    for paragraph in paragraphs:
        lines = paragraph.split('\n', 1)
        if len(lines) > 1:
            processed_paragraphs.append(lines[1])
        else:
            # If paragraph is only one line, it gets removed entirely
            continue

    # Join processed paragraphs and split into sentences
    full_text = '\n\n'.join(processed_paragraphs)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', full_text)

    # Write each sentence to the output file, each on a new line
    with open(output_path, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            # Normalize spacing after punctuation
            sentence = re.sub(r'([,.!?)\'(\[/-;-"])', r' \1 ', sentence)  # Add space after these punctuation marks
            sentence = re.sub(r'\s+', ' ', sentence)  # Remove multiple spaces if they exist
            sentence = sentence.strip()  # Strip leading and trailing spaces
            file.write(sentence + '\n')

if __name__ == "__main__":

    corpus_path = "data/corpus.txt"
    processed_corpus_path = "data/processed_corpus.txt"
    vocab_path = "data/corpus_vocab.txt" 
    
    process_file(corpus_path, processed_corpus_path)

    vocab = Vocabulary(processed_corpus_path, min_occurrences=500, build=True)
    vocab.write_vocab_file(vocab_path)