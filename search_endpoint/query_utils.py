import re
from typing import Set


def normalize_text(text: str) -> str:
        """Normalize line breaks and whitespace globally"""
        # Replace all line breaks (including \r\n) with single space
        text = re.sub(r'[\r\n]+', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
class QueryProcessor:
    """Query processing utilities"""
    
    STOP_WORDS: Set[str] = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    @classmethod
    

    def clean_query(cls, raw_query: str) -> str:
        """Clean and normalize user query"""
        try:
            cleaned = cls.normalize_text(raw_query)
            # Remove extra whitespace
            # cleaned = re.sub(r'\s+', ' ', raw_query.strip())
            
            # Remove special characters that might interfere with search
            cleaned = re.sub(r'[^\w\s\-\.\?\!]', '', cleaned)
            
            # Convert to lowercase for consistency
            cleaned = cleaned.lower()
            
            # Remove common stop words that don't add semantic value for search
            words = cleaned.split()
            cleaned_words = [word for word in words if word not in cls.STOP_WORDS or len(words) <= 3]
            cleaned = ' '.join(cleaned_words)
            
            return cleaned
            
        except Exception:
            return raw_query  # Return original if cleaning fails