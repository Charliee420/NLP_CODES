# Text Preprocessing for NLP

A comprehensive guide and implementation of Natural Language Processing (NLP) text preprocessing techniques using Python.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing Steps](#preprocessing-steps)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Output Files](#output-files)
- [Preprocessing Techniques Explained](#preprocessing-techniques-explained)
- [Results and Analysis](#results-and-analysis)
- [Requirements](#requirements)
- [Contributing](#contributing)

---

## 🎯 Overview

This project demonstrates essential **text preprocessing techniques** used in Natural Language Processing. Text preprocessing is a crucial step in any NLP pipeline that transforms raw text into a clean, structured format suitable for machine learning models and analysis.

The project uses a paragraph about **emojis** extracted from Wikipedia as the sample dataset and applies various preprocessing techniques including:

- Text normalization
- Tokenization
- Stopword removal
- Stemming and Lemmatization
- Frequency analysis

---

## 📊 Dataset

**Source:** [Wikipedia - Emoji](https://en.wikipedia.org/wiki/Emoji)

**File:** `text.txt`

The dataset is a comprehensive paragraph explaining the origin, purpose, and cultural significance of emojis. This text contains:
- Multiple sentences
- Various punctuation marks
- Emoji characters
- Technical terminology
- References to different languages (Japanese)

---

## 🔧 Preprocessing Steps

The preprocessing pipeline consists of the following steps:

### 1. **Text Loading**
   - Read the text from `text.txt`
   - Display original text and character count

### 2. **Lowercase Conversion**
   - Convert all text to lowercase
   - Purpose: Standardize text and reduce vocabulary size

### 3. **Emoji Removal**
   - Remove all emoji characters and special Unicode symbols
   - Purpose: Clean text for analysis (emojis are removed from the emoji description text)

### 4. **Punctuation Removal**
   - Remove all punctuation marks (.,!?;: etc.)
   - Purpose: Focus on actual words

### 5. **Whitespace Normalization**
   - Remove extra spaces, tabs, and newlines
   - Purpose: Clean and standardize text format

### 6. **Word Tokenization**
   - Split text into individual words (tokens)
   - Uses NLTK's `word_tokenize()` or simple split as fallback

### 7. **Sentence Tokenization**
   - Split text into individual sentences
   - Uses NLTK's `sent_tokenize()` or regex-based split as fallback

### 8. **Stopword Removal**
   - Remove common words that don't add semantic value
   - Examples: "the", "is", "and", "or", "but"
   - Purpose: Focus on meaningful words

### 9. **Stemming**
   - Reduce words to their root form using Porter Stemmer
   - Example: "running" → "run", "pictures" → "pictur"
   - Purpose: Group similar words together

### 10. **Lemmatization**
   - Convert words to their dictionary form using WordNet Lemmatizer
   - Example: "better" → "good", "running" → "run"
   - Purpose: More accurate word normalization than stemming

### 11. **Word Frequency Analysis**
   - Count occurrence of each word
   - Display top 10 most common words
   - Calculate vocabulary statistics

### 12. **Save Results**
   - Save preprocessed text to `preprocessed_text.txt`
   - Save tokens to `tokens.txt`

---

## 💻 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone or Download this Repository

```bash
cd Text-Preprocessing-NLP
```

### Step 2: Install Required Packages

```bash
pip install nltk
```

### Step 3: Download NLTK Data

The script will automatically download required NLTK data on first run, but you can also do it manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

---

## 🚀 Usage

### Running the Script

```bash
python text_preprocessing.py
```

### Expected Output

The script will display:
1. Original text
2. Each preprocessing step with results
3. Summary statistics
4. Saved file locations

### Example Output:

```
================================================================================
  1. ORIGINAL TEXT
================================================================================
An emoji is a pictogram, logogram, ideogram, or smiley embedded in text...
Length: 710 characters

================================================================================
  2. LOWERCASE CONVERSION
================================================================================
an emoji is a pictogram, logogram, ideogram, or smiley embedded in text...

================================================================================
  11. WORD FREQUENCY ANALYSIS
================================================================================
Top 10 most common words:
  emoji: 5
  text: 3
  pictures: 2
  ...
```

---

## 📁 Code Structure

```
Text-Preprocessing-NLP/
│
├── text.txt                    # Original text (input)
├── text_preprocessing.py       # Main preprocessing script
├── README.md                   # This documentation file
├── preprocessed_text.txt       # Output: Cleaned text
└── tokens.txt                  # Output: Word tokens
```

### Class: `TextPreprocessor`

A comprehensive class containing all preprocessing methods:

```python
class TextPreprocessor:
    def read_text(filepath)              # Read text from file
    def lowercase_text(text)             # Convert to lowercase
    def remove_punctuation(text)         # Remove punctuation
    def remove_numbers(text)             # Remove numbers
    def remove_whitespace(text)          # Clean whitespace
    def remove_emoji(text)               # Remove emoji/special chars
    def tokenize_words(text)             # Split into words
    def tokenize_sentences(text)         # Split into sentences
    def remove_stopwords(tokens)         # Remove common words
    def stem_words(tokens)               # Apply stemming
    def lemmatize_words(tokens)          # Apply lemmatization
    def get_word_frequency(tokens)       # Count word frequencies
    def preprocess_pipeline(text, steps) # Run full pipeline
```

---

## 📄 Output Files

### 1. `preprocessed_text.txt`
Contains the fully preprocessed text after all cleaning steps:
- Lowercase
- No emoji
- No punctuation
- No extra whitespace

### 2. `tokens.txt`
Contains individual word tokens (one per line) after:
- Full preprocessing
- Stopword removal

---

## 📚 Preprocessing Techniques Explained

### 🔤 Lowercase Conversion

**Why?** Machine learning models treat "Apple" and "apple" as different words. Lowercasing ensures uniformity.

**Example:**
```
Before: "An emoji is a pictogram"
After:  "an emoji is a pictogram"
```

### 🧹 Punctuation Removal

**Why?** Punctuation can interfere with word matching and doesn't usually add semantic value.

**Example:**
```
Before: "emoji, but emoji are pictures!"
After:  "emoji but emoji are pictures"
```

### 📝 Tokenization

**Why?** Breaks text into processable units (words or sentences).

**Word Tokenization:**
```
Input: "emoji are pictures"
Output: ["emoji", "are", "pictures"]
```

**Sentence Tokenization:**
```
Input: "First sentence. Second sentence!"
Output: ["First sentence.", "Second sentence!"]
```

### 🚫 Stopword Removal

**Why?** Removes common words that don't contribute to meaning.

**Example:**
```
Before: ["the", "emoji", "is", "a", "picture"]
After:  ["emoji", "picture"]
```

**Common Stopwords:** the, is, a, an, and, or, but, in, on, at, to, for, of, as, by, was, are

### 🌱 Stemming

**Why?** Groups words with the same root together.

**Algorithm:** Porter Stemmer

**Examples:**
- running → run
- runner → runner (sometimes imperfect)
- pictures → pictur

**Note:** Stemming is fast but sometimes produces non-words.

### 🔍 Lemmatization

**Why?** More accurate than stemming; produces valid words.

**Algorithm:** WordNet Lemmatizer

**Examples:**
- running → run
- better → good
- are → be

**Note:** Lemmatization is slower but more accurate than stemming.

### 📊 Frequency Analysis

**Why?** Understand word distribution and importance.

**Metrics:**
- **Word Count:** Total number of words
- **Unique Words:** Number of distinct words
- **Vocabulary Richness:** unique_words / total_words
- **Most Common Words:** Top N most frequent words

---

## 📈 Results and Analysis

### Sample Statistics (for the emoji text):

| Metric | Value |
|--------|-------|
| Original text length | ~710 characters |
| Number of sentences | 7 |
| Total words (tokens) | ~110 |
| Words after stopword removal | ~60 |
| Unique words | ~55 |
| Vocabulary richness | ~90% |

### Top Words (Example):
1. emoji - 5 occurrences
2. pictures - 3 occurrences
3. text - 2 occurrences
4. embedded - 2 occurrences

---

## 🛠️ Requirements

```txt
nltk==3.8.1
```

### Optional Dependencies:
- `re` (built-in)
- `string` (built-in)
- `collections` (built-in)

---

## 🎓 Learning Outcomes

After working through this project, you will understand:

1. ✅ **Why preprocessing is essential** in NLP pipelines
2. ✅ **How to clean and normalize text** data
3. ✅ **Different tokenization strategies** (word vs. sentence)
4. ✅ **Stopword removal** and its impact
5. ✅ **Difference between stemming and lemmatization**
6. ✅ **Text frequency analysis** techniques
7. ✅ **Working with NLTK** library
8. ✅ **Building a reusable preprocessing pipeline**

---

## 🔜 Next Steps

To extend this project, consider:

1. **Add More Preprocessing:**
   - HTML tag removal
   - URL removal
   - Hashtag/mention handling
   - Spell correction

2. **Visualization:**
   - Word clouds
   - Frequency histograms
   - N-gram analysis

3. **Advanced NLP:**
   - Part-of-speech tagging
   - Named Entity Recognition (NER)
   - Dependency parsing
   - Sentiment analysis

4. **Batch Processing:**
   - Process multiple documents
   - Create a corpus preprocessing pipeline

---

## 🤝 Contributing

Feel free to:
- Add new preprocessing techniques
- Improve existing methods
- Add more example datasets
- Enhance documentation

---

## 📝 Notes

- The script works with or without NLTK (graceful fallback)
- Emoji removal is comprehensive, covering most Unicode ranges
- Frequency analysis helps identify key topics in the text
- The class-based structure makes it easy to reuse in other projects

---

## 📞 Support

For questions or issues:
1. Review the code comments
2. Check NLTK documentation: https://www.nltk.org/
3. Review preprocessing concepts in NLP literature

---

## 🏆 Acknowledgments

- **Wikipedia** for the emoji article
- **NLTK** team for the excellent NLP library
- **Porter Stemmer** algorithm creator
- **WordNet** lexical database

---

## 📜 License

This project is for educational purposes.

---

**Happy Learning! 🚀**

---

### Quick Reference

```python
# Import the preprocessor
from text_preprocessing import TextPreprocessor

# Initialize
preprocessor = TextPreprocessor()

# Read text
text = preprocessor.read_text('your_file.txt')

# Preprocess
clean_text = preprocessor.preprocess_pipeline(text)

# Tokenize
tokens = preprocessor.tokenize_words(clean_text)

# Remove stopwords
filtered = preprocessor.remove_stopwords(tokens)

# Get insights
frequency = preprocessor.get_word_frequency(filtered)
```

---

*Last Updated: February 14, 2026*
