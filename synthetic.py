import numpy as np
from typing import List, Tuple, Dict
from transformers import pipeline
from nltk import ngrams
import spacy
from collections import defaultdict

class EnhancedPatternBasedGenerator:
    def __init__(self):
        # Load models and tools
        self.nlp = spacy.load("en_core_web_sm")
        self.en_fr_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
        self.fr_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
        
        # Storage for patterns
        self.ngram_patterns = defaultdict(lambda: defaultdict(int))
        self.context_patterns = defaultdict(list)
        self.error_frequencies = defaultdict(int)
        self.pos_error_patterns = defaultdict(lambda: defaultdict(int))
        
        # Add storage for training data
        self.corrected_texts = []
        self.raw_texts = []
        
    def back_translate(self, text: str) -> str:
        """Create natural variations using back-translation."""
        try:
            # English -> French with increased max_length
            fr_text = self.en_fr_translator(text, max_length=1024, truncation=True)[0]['translation_text']
            # French -> English with increased max_length
            en_text = self.fr_en_translator(fr_text, max_length=1024, truncation=True)[0]['translation_text']
            return en_text
        except:
            return text
    
    def learn_ngram_patterns(self, raw_texts: List[str], corrected_texts: List[str]):
        """Learn n-gram based error patterns."""
        for raw, correct in zip(raw_texts, corrected_texts):
            raw_doc = self.nlp(raw)
            correct_doc = self.nlp(correct)
            
            # Learn 1-3 gram patterns
            for n in range(1, 4):
                raw_ngrams = list(ngrams(raw_doc, n))
                correct_ngrams = list(ngrams(correct_doc, n))
                
                for r_gram, c_gram in zip(raw_ngrams, correct_ngrams):
                    if r_gram != c_gram:
                        self.ngram_patterns[' '.join([t.text for t in c_gram])][' '.join([t.text for t in r_gram])] += 1

    def learn_pos_patterns(self, raw_texts: List[str], corrected_texts: List[str]):
        """Learn POS-based error patterns."""
        for raw, correct in zip(raw_texts, corrected_texts):
            raw_doc = self.nlp(raw)
            correct_doc = self.nlp(correct)
            
            for raw_token, correct_token in zip(raw_doc, correct_doc):
                if raw_token.text != correct_token.text:
                    self.pos_error_patterns[correct_token.pos_][
                        (raw_token.text, correct_token.text)
                    ] += 1

    def learn_patterns(self, raw_texts: List[str], corrected_texts: List[str]):
        """Learn all patterns from the training data."""
        # Store the training data
        self.raw_texts = raw_texts
        self.corrected_texts = corrected_texts
        
        self.learn_ngram_patterns(raw_texts, corrected_texts)
        self.learn_pos_patterns(raw_texts, corrected_texts)
        
        # Learn context patterns
        for raw, correct in zip(raw_texts, corrected_texts):
            raw_doc = self.nlp(raw)
            correct_doc = self.nlp(correct)
            
            for i, (raw_token, correct_token) in enumerate(zip(raw_doc, correct_doc)):
                if raw_token.text != correct_token.text:
                    context_before = ' '.join([t.text for t in raw_doc[max(0, i-2):i]])
                    context_after = ' '.join([t.text for t in raw_doc[i+1:min(len(raw_doc), i+3)]])
                    
                    self.context_patterns[(raw_token.text, correct_token.text)].append(
                        (context_before, context_after)
                    )
                    self.error_frequencies[(raw_token.text, correct_token.text)] += 1

    def generate_error(self, word: str, pos: str, context_before: str, context_after: str) -> str:
        """Generate context-aware errors."""
        # Try POS-based errors first
        pos_errors = self.pos_error_patterns[pos]
        if pos_errors:
            # Convert the dictionary items into separate lists for errors and frequencies
            error_pairs = list(pos_errors.items())
            errors = [pair[0][0] for pair in error_pairs]
            freqs = [float(pair[1]) for pair in error_pairs]  # Convert to float explicitly
            
            total = sum(freqs)
            if total > 0:
                # Ensure probabilities sum to 1 and are all positive
                probs = np.array(freqs, dtype=float) / total
                if np.all(probs >= 0) and np.isclose(np.sum(probs), 1):
                    return np.random.choice(errors, p=probs)
                
            # Fallback to random choice if probabilities are invalid
            return np.random.choice(errors)
        
        # Fall back to context-based errors
        possible_errors = []
        for (error_word, correct_word), contexts in self.context_patterns.items():
            if correct_word.lower() == word.lower():
                possible_errors.append(error_word)
        
        if possible_errors:
            frequencies = [self.error_frequencies[(e, word)] for e in possible_errors]
            total_freq = sum(frequencies)
            if total_freq > 0:
                probabilities = np.array(frequencies) / total_freq
                return np.random.choice(possible_errors, p=probabilities)
        
        return word

    def generate_sample(self) -> Tuple[str, str]:
        """Generate a single synthetic sample."""
        # Convert numpy.str_ to Python str when selecting the random text
        selected_text = str(np.random.choice(self.corrected_texts))
        correct_doc = self.nlp(self.back_translate(selected_text))
        
        # Generate errors while preserving structure
        raw_tokens = []
        for i, token in enumerate(correct_doc):
            context_before = ' '.join([t.text for t in correct_doc[max(0, i-2):i]])
            context_after = ' '.join([t.text for t in correct_doc[i+1:min(len(correct_doc), i+3)]])
            
            # Decide whether to introduce an error (20% chance)
            if np.random.random() < 0.2:
                error_token = self.generate_error(token.text, token.pos_, context_before, context_after)
                raw_tokens.append(error_token)
            else:
                raw_tokens.append(token.text)
        
        raw_text = ' '.join(raw_tokens)
        correct_text = str(correct_doc)
        
        return raw_text, correct_text

    def generate_dataset(self, num_samples: int) -> Tuple[List[str], List[str]]:
        """Generate multiple samples in batches."""
        batch_size = 2500
        all_raw_texts = []
        all_corrected_texts = []
        
        for i in range(0, num_samples, batch_size):
            batch_count = min(batch_size, num_samples - i)
            batch_samples = [self.generate_sample() for _ in range(batch_count)]
            
            raw_batch, corrected_batch = zip(*batch_samples)
            all_raw_texts.extend(raw_batch)
            all_corrected_texts.extend(corrected_batch)
            
            print(f"Generated {len(all_raw_texts)} samples out of {num_samples}")
        
        return all_raw_texts, all_corrected_texts

def save_dataset(raw_texts: List[str], corrected_texts: List[str], output_file: str):
    """Save the synthetic dataset to a CSV file."""
    import pandas as pd
    
    df = pd.DataFrame({
        'raw_text': raw_texts,
        'corrected_text': corrected_texts
    })
    
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

def load_real_data(file_path: str) -> Tuple[List[str], List[str]]:
    """Load raw and corrected texts from a CSV file."""
    import pandas as pd
    df = pd.DataFrame(pd.read_csv(file_path))
    return df['raw_text'].tolist(), df['corrected_text'].tolist()

if __name__ == "__main__":
    # Load real data
    raw_texts, corrected_texts = load_real_data("real_data.csv")
    
    # Initialize and train generator
    generator = EnhancedPatternBasedGenerator()
    generator.learn_patterns(raw_texts, corrected_texts)
    
    # Generate new samples
    synthetic_raw_texts, synthetic_corrected_texts = generator.generate_dataset(num_samples=5000)
    
    # Save the synthetic dataset
    output_file = "synthetic_dataset_enhanced.csv"
    save_dataset(synthetic_raw_texts, synthetic_corrected_texts, output_file)