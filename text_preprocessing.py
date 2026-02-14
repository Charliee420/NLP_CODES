"""
Text Preprocessing for NLP
This script demonstrates various NLP preprocessing techniques on the text.txt file.
"""

import re
import string
from collections import Counter

# Try to import NLTK components
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not installed. Some features will be unavailable.")
    print("Install with: pip install nltk")


class TextPreprocessor:
    """A comprehensive text preprocessing class for NLP tasks."""
    
    def __init__(self):
        """Initialize the preprocessor with necessary components."""
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        
    def read_text(self, filepath):
        """Read text from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def lowercase_text(self, text):
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_punctuation(self, text):
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_numbers(self, text):
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
    
    def remove_whitespace(self, text):
        """Remove extra whitespace from text."""
        return ' '.join(text.split())
    
    def remove_emoji(self, text):
        """Remove emoji and special characters from text."""
        # Remove emoji
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove other special characters but keep basic punctuation and spaces
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text
    
    def tokenize_words(self, text):
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            return word_tokenize(text)
        else:
            # Simple tokenization fallback
            return text.split()
    
    def tokenize_sentences(self, text):
        """Tokenize text into sentences."""
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        else:
            # Simple sentence splitting fallback
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens."""
        if NLTK_AVAILABLE:
            stop_words = set(stopwords.words('english'))
            return [word for word in tokens if word.lower() not in stop_words]
        else:
            # Basic stopwords list as fallback
            basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                             'at', 'to', 'for', 'of', 'as', 'by', 'is', 'are', 'was'}
            return [word for word in tokens if word.lower() not in basic_stopwords]
    
    def stem_words(self, tokens):
        """Apply stemming to tokens."""
        if self.stemmer:
            return [self.stemmer.stem(word) for word in tokens]
        else:
            print("Stemming unavailable without NLTK")
            return tokens
    
    def lemmatize_words(self, tokens):
        """Apply lemmatization to tokens."""
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(word) for word in tokens]
        else:
            print("Lemmatization unavailable without NLTK")
            return tokens
    
    def get_word_frequency(self, tokens):
        """Get word frequency distribution."""
        return Counter(tokens)
    
    def preprocess_pipeline(self, text, steps=None):
        """
        Run a complete preprocessing pipeline.
        
        Args:
            text: Input text to preprocess
            steps: List of preprocessing steps to apply
                   Available: 'lowercase', 'remove_emoji', 'remove_punctuation', 
                             'remove_numbers', 'remove_whitespace'
        
        Returns:
            Preprocessed text
        """
        if steps is None:
            steps = ['lowercase', 'remove_emoji', 'remove_punctuation', 
                    'remove_numbers', 'remove_whitespace']
        
        processed_text = text
        
        for step in steps:
            if step == 'lowercase':
                processed_text = self.lowercase_text(processed_text)
            elif step == 'remove_emoji':
                processed_text = self.remove_emoji(processed_text)
            elif step == 'remove_punctuation':
                processed_text = self.remove_punctuation(processed_text)
            elif step == 'remove_numbers':
                processed_text = self.remove_numbers(processed_text)
            elif step == 'remove_whitespace':
                processed_text = self.remove_whitespace(processed_text)
        
        return processed_text
    
    def print_separator(self, title):
        """Print a formatted separator."""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)


def main():
    """Main function to demonstrate text preprocessing."""
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Read the text file
    preprocessor.print_separator("1. ORIGINAL TEXT")
    text = preprocessor.read_text('text.txt')
    
    if text is None:
        return
    
    print(text)
    print(f"\nLength: {len(text)} characters")
    
    # Step 1: Lowercase conversion
    preprocessor.print_separator("2. LOWERCASE CONVERSION")
    lowercased = preprocessor.lowercase_text(text)
    print(lowercased)
    
    # Step 2: Remove emoji
    preprocessor.print_separator("3. REMOVE EMOJI")
    no_emoji = preprocessor.remove_emoji(lowercased)
    print(no_emoji)
    
    # Step 3: Remove punctuation
    preprocessor.print_separator("4. REMOVE PUNCTUATION")
    no_punct = preprocessor.remove_punctuation(no_emoji)
    print(no_punct)
    
    # Step 4: Remove extra whitespace
    preprocessor.print_separator("5. REMOVE EXTRA WHITESPACE")
    clean_text = preprocessor.remove_whitespace(no_punct)
    print(clean_text)
    
    # Step 5: Tokenization (words)
    preprocessor.print_separator("6. WORD TOKENIZATION")
    tokens = preprocessor.tokenize_words(clean_text)
    print(f"Tokens: {tokens[:20]}...")  # Show first 20 tokens
    print(f"Total tokens: {len(tokens)}")
    
    # Step 6: Sentence tokenization (using original text)
    preprocessor.print_separator("7. SENTENCE TOKENIZATION")
    sentences = preprocessor.tokenize_sentences(text)
    print(f"Number of sentences: {len(sentences)}")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")
    
    # Step 7: Remove stopwords
    preprocessor.print_separator("8. REMOVE STOPWORDS")
    no_stopwords = preprocessor.remove_stopwords(tokens)
    print(f"Tokens after stopword removal: {no_stopwords[:20]}...")
    print(f"Total tokens after stopword removal: {len(no_stopwords)}")
    print(f"Stopwords removed: {len(tokens) - len(no_stopwords)}")
    
    # Step 8: Stemming
    preprocessor.print_separator("9. STEMMING")
    stemmed = preprocessor.stem_words(no_stopwords)
    print(f"Stemmed tokens: {stemmed[:20]}...")
    
    # Step 9: Lemmatization
    preprocessor.print_separator("10. LEMMATIZATION")
    lemmatized = preprocessor.lemmatize_words(no_stopwords)
    print(f"Lemmatized tokens: {lemmatized[:20]}...")
    
    # Step 10: Word frequency
    preprocessor.print_separator("11. WORD FREQUENCY ANALYSIS")
    freq = preprocessor.get_word_frequency(no_stopwords)
    print("Top 10 most common words:")
    for word, count in freq.most_common(10):
        print(f"  {word}: {count}")
    
    # Summary statistics
    preprocessor.print_separator("12. SUMMARY STATISTICS")
    print(f"Original text length: {len(text)} characters")
    print(f"Number of sentences: {len(sentences)}")
    print(f"Total words (tokens): {len(tokens)}")
    print(f"Words after stopword removal: {len(no_stopwords)}")
    print(f"Unique words: {len(set(no_stopwords))}")
    print(f"Vocabulary richness: {len(set(no_stopwords)) / len(no_stopwords):.2%}")
    
    # Save preprocessed text
    preprocessor.print_separator("13. SAVING PREPROCESSED TEXT")
    with open('preprocessed_text.txt', 'w', encoding='utf-8') as f:
        f.write(clean_text)
    print("Preprocessed text saved to: preprocessed_text.txt")
    
    # Save tokens
    with open('tokens.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(no_stopwords))
    print("Tokens (without stopwords) saved to: tokens.txt")
    
    print("\n" + "="*80)
    print("  PREPROCESSING COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Download required NLTK data if available
    if NLTK_AVAILABLE:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except:
            pass
    
    main()
