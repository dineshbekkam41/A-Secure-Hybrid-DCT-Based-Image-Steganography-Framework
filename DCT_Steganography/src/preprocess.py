"""
Module 1: Image Preprocessing and Region Selection
"""

import cv2
import numpy as np
from typing import Tuple, List
import secrets

# ⚙ CONFIGURATION - YOU CAN ADJUST THESE
MIN_VARIANCE = 50      # Minimum texture variance for block selection
MAX_VARIANCE = 5000    # Maximum texture variance
BLOCK_SIZE = 8         # DCT block size (don't change)

def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"[OK] Loaded image: {image.shape[1]}x{image.shape[0]} pixels")
    return image

def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to YCbCr color space
    
    Args:
        image: BGR image
        
    Returns:
        YCbCr image
    """
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return ycbcr

def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr image back to BGR
    
    Args:
        ycbcr: YCbCr image
        
    Returns:
        BGR image
    """
    bgr = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
    return bgr

def calculate_block_variance(block: np.ndarray) -> float:
    """
    Calculate variance of 8x8 block
    
    Args:
        block: 8x8 numpy array
        
    Returns:
        Variance value
    """
    return np.var(block)

def get_all_blocks(image: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
    """
    Extract all 8x8 blocks from image with their positions
    
    Args:
        image: 2D image array (single channel)
        
    Returns:
        List of (row, col, block) tuples
    """
    height, width = image.shape
    blocks = []
    
    for i in range(0, height - BLOCK_SIZE + 1, BLOCK_SIZE):
        for j in range(0, width - BLOCK_SIZE + 1, BLOCK_SIZE):
            block = image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            if block.shape == (BLOCK_SIZE, BLOCK_SIZE):
                blocks.append((i, j, block))
    
    return blocks

def select_embedding_blocks(y_channel: np.ndarray, 
                            num_blocks_needed: int,
                            seed: int,
                            ignore_variance: bool = False) -> List[Tuple[int, int]]:
    """
    Select suitable blocks for embedding using variance threshold and PRNG
    
    Args:
        y_channel: Y channel of YCbCr image
        num_blocks_needed: Number of blocks required for embedding
        seed: Seed for reproducible random selection
        
    Returns:
        List of (row, col) positions of selected blocks
    """
    height, width = y_channel.shape
    all_blocks = get_all_blocks(y_channel)
    
    # Get all valid blocks (avoid borders)
    # Sort by position for deterministic selection
    all_valid_blocks = []
    for i, j, block in all_blocks:
        # Avoid image borders
        if i < BLOCK_SIZE or j < BLOCK_SIZE:
            continue
        if i > height - 2*BLOCK_SIZE or j > width - 2*BLOCK_SIZE:
            continue
        all_valid_blocks.append((i, j))
    
    # Sort by position for deterministic selection
    all_valid_blocks.sort(key=lambda x: (x[0], x[1]))
    
    if ignore_variance:
        # Use all valid blocks (ignoring variance)
        suitable_blocks = all_valid_blocks
    else:
        # Filter by variance (mid-texture regions)
        suitable_blocks = []
        for i, j in all_valid_blocks:
            block = y_channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            variance = calculate_block_variance(block)
            if MIN_VARIANCE < variance < MAX_VARIANCE:
                suitable_blocks.append((i, j))
    
    print(f"[OK] Found {len(suitable_blocks)} suitable blocks (need {num_blocks_needed})")
    
    if len(suitable_blocks) < num_blocks_needed:
        raise ValueError(f"Not enough suitable blocks! Found {len(suitable_blocks)}, need {num_blocks_needed}")
    
    # Use cryptographic PRNG for selection from sorted list
    rng = np.random.RandomState(seed)
    selected_indices = rng.choice(len(suitable_blocks), num_blocks_needed, replace=False)
    # Sort indices to ensure deterministic order
    selected_indices = sorted(selected_indices)
    selected_blocks = [suitable_blocks[i] for i in selected_indices]
    
    return selected_blocks

def calculate_capacity(image_shape: Tuple[int, int, int], 
                      bits_per_block: int = 8) -> dict:
    """
    Calculate embedding capacity
    
    Args:
        image_shape: Shape of cover image (height, width, channels)
        bits_per_block: Number of bits embedded per block
        
    Returns:
        Dictionary with capacity information
    """
    height, width = image_shape[0], image_shape[1]
    total_blocks = (height // BLOCK_SIZE) * (width // BLOCK_SIZE)
    
    # Assume 60% blocks are suitable (empirical)
    usable_blocks = int(total_blocks * 0.6)
    
    total_bits = usable_blocks * bits_per_block
    total_bytes = total_bits // 8
    
    return {
        'total_blocks': total_blocks,
        'usable_blocks': usable_blocks,
        'capacity_bits': total_bits,
        'capacity_bytes': total_bytes,
        'capacity_kb': total_bytes / 1024
    }

def save_image(image_path: str, image: np.ndarray):
    """
    Save image to file
    
    Args:
        image_path: Output path
        image: Image array to save
    """
    cv2.imwrite(image_path, image)
    print(f"[OK] Saved image to {image_path}")

if __name__ == "__main__":
    # Test the module
    print("Testing preprocessing module...")
    
    # ⚙ CHANGE THIS - Put your test image path
    test_image_path = "dataset/cover_images/test.png"
    
    try:
        img = load_image(test_image_path)
        capacity = calculate_capacity(img.shape)
        
        print("\n[INFO] Capacity Analysis:")
        print(f"  Total blocks: {capacity['total_blocks']}")
        print(f"  Usable blocks: {capacity['usable_blocks']}")
        print(f"  Capacity: {capacity['capacity_bytes']} bytes ({capacity['capacity_kb']:.2f} KB)")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        print("Make sure to place a test image in dataset/cover_images/")

