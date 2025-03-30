"""Image comparison module supporting multiple comparison methods."""

import cv2
import numpy as np
from typing import Tuple, List, Dict
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Cache for storing computed hashes
_hash_cache: Dict[int, List[bool]] = {}

@lru_cache(maxsize=128)
def to_gray(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Convert image to grayscale and resize."""
    return cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2GRAY)

def hamming_distance(hash1: List[bool], hash2: List[bool]) -> float:
    """Calculate normalized Hamming distance between two hashes."""
    return sum(h1 != h2 for h1, h2 in zip(hash1, hash2)) / len(hash1)

def compute_dhash(image: np.ndarray) -> List[bool]:
    """Compute difference hash for an image."""
    # Use image data pointer as cache key
    cache_key = image.__array_interface__['data'][0]
    if cache_key in _hash_cache:
        return _hash_cache[cache_key]
    
    # Resize to 9x8 and convert to grayscale
    resized = to_gray(image, (9, 8))
    
    # Calculate difference and build hash
    diff = resized[:, 1:] > resized[:, :-1]  # 8x8 boolean array
    hash_value = diff.flatten().tolist()  # 64 bits hash
    
    # Cache the result
    _hash_cache[cache_key] = hash_value
    return hash_value

def dhash(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute dHash and compare two images.
    
    Steps:
    1. Resize to 9x8 (72 pixels)
    2. Convert to grayscale
    3. Compute difference hash (compare adjacent pixels)
    4. Compare hashes using Hamming distance
    """
    return hamming_distance(compute_dhash(img1), compute_dhash(img2))

def compute_phash(image: np.ndarray) -> List[bool]:
    """Compute perceptual hash for an image."""
    # Use image data pointer as cache key
    cache_key = image.__array_interface__['data'][0]
    if cache_key in _hash_cache:
        return _hash_cache[cache_key]
    
    size = (32, 32)
    gray = to_gray(image, size)
    dct = cv2.dct(np.float32(gray))[:8, :8]
    hash_value = (dct > np.mean(dct)).flatten().tolist()
    
    # Cache the result
    _hash_cache[cache_key] = hash_value
    return hash_value

def phash(img1: np.ndarray, img2: np.ndarray) -> float:
    """pHash comparison."""
    return hamming_distance(compute_phash(img1), compute_phash(img2))

def histogram(img1: np.ndarray, img2: np.ndarray) -> float:
    """Histogram comparison."""
    size = (256, 256)
    img1_resized = cv2.resize(img1, size)
    img2_resized = cv2.resize(img2, size)
    
    def get_channel_similarity(ch1: np.ndarray, ch2: np.ndarray) -> float:
        hist1 = cv2.calcHist([ch1], [0], None, [256], [0.0, 255.0])
        hist2 = cv2.calcHist([ch2], [0], None, [256], [0.0, 255.0])
        
        mask = hist1 != hist2
        max_vals = np.maximum(hist1, hist2)
        similarity = np.ones_like(hist1)
        similarity[mask] = 1 - np.abs(hist1[mask] - hist2[mask]) / max_vals[mask]
        return float(np.mean(similarity))
    
    channels1, channels2 = cv2.split(img1_resized), cv2.split(img2_resized)
    similarities = [get_channel_similarity(ch1, ch2) 
                   for ch1, ch2 in zip(channels1, channels2)]
    return sum(similarities) / len(similarities)

_METHODS = {
    'dhash': dhash,
    'phash': phash,
    'histogram': histogram
}

def compare_images(img1: np.ndarray, img2: np.ndarray, method: str = "histogram") -> float:
    """
    Compare two images using specified method.
    
    Args:
        img1, img2: Images to compare
        method: Comparison method ('dhash', 'phash', or 'histogram')
        
    Returns:
        float: Similarity score in range [0, 1]
              For hash methods (dhash, phash): lower = more similar
              For histogram: higher = more similar
    """
    try:
        return _METHODS[method.lower()](img1, img2)
    except KeyError:
        logger.warning(f"Invalid method '{method}', using dhash")
        return dhash(img1, img2)
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise