import numpy as np
from typing import List, Tuple

from src.utils.nlp import tokenize, bag_of_words


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def find_best_response(
    user_input: str,
    conversations: List[dict],
    all_words: List[str],
    top_k: int = 1,
) -> Tuple[str, float]:
    """
    Find the best response from conversations by similarity matching.
    
    Args:
        user_input: User's query
        conversations: List of {"input": str, "output": str} dicts
        all_words: Vocabulary for bag-of-words
        top_k: Return top-k matches (currently returns best one)
    
    Returns:
        (response, confidence) where confidence is the similarity score
    """
    # Tokenize user input and convert to bag-of-words
    user_tokens = tokenize(user_input)
    user_bow = bag_of_words(user_tokens, all_words)
    
    best_score = -1.0
    best_response = None
    
    # Find the conversation with highest similarity
    for conv in conversations:
        conv_tokens = tokenize(conv["input"])
        conv_bow = bag_of_words(conv_tokens, all_words)
        
        score = cosine_similarity(user_bow, conv_bow)
        if score > best_score:
            best_score = score
            best_response = conv["output"]
    
    return best_response if best_response else "Sorry, I couldn't understand that.", best_score


def retrieve_top_k_responses(
    user_input: str,
    conversations: List[dict],
    all_words: List[str],
    k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k similar responses.
    
    Returns:
        List of (response, similarity_score) tuples, sorted by score descending
    """
    user_tokens = tokenize(user_input)
    user_bow = bag_of_words(user_tokens, all_words)
    
    scores = []
    for conv in conversations:
        conv_tokens = tokenize(conv["input"])
        conv_bow = bag_of_words(conv_tokens, all_words)
        
        score = cosine_similarity(user_bow, conv_bow)
        scores.append((conv["output"], score))
    
    # Sort by score descending and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
