"""
Module 3: DCT Embedding and Extraction Engine
"""

import cv2
import numpy as np
from typing import List, Tuple
from src.preprocess import (
    rgb_to_ycbcr, ycbcr_to_rgb, select_embedding_blocks,
    BLOCK_SIZE
)

# ⚙ CONFIGURATION
# DCT coefficient positions for embedding (mid-frequency, zigzag order)
# These positions are robust to JPEG compression
EMBED_POSITIONS = [
    (1, 2), (2, 1), (2, 2), (1, 3),
    (3, 1), (3, 2), (2, 3), (3, 3)
]  # 8 positions per block = 8 bits per block

def derive_seed_from_password(password: str) -> int:
    """
    Derive reproducible seed from password for PRNG
    
    Args:
        password: User password
        
    Returns:
        Integer seed
    """
    import hashlib
    hash_obj = hashlib.sha256(password.encode())
    seed = int.from_bytes(hash_obj.digest()[:4], 'big')
    return seed

def bytes_to_bits(data: bytes) -> List[int]:
    """
    Convert bytes to list of bits
    
    Args:
        data: Input bytes
        
    Returns:
        List of bits (0 or 1)
    """
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits

def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Convert list of bits to bytes
    
    Args:
        bits: List of bits
        
    Returns:
        Bytes object
    """
    # Pad to multiple of 8
    while len(bits) % 8 != 0:
        bits.append(0)
    
    byte_array = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | bits[i + j]
        byte_array.append(byte_val)
    
    return bytes(byte_array)

def quantization_index_modulation(coef: float, bit: int, k: int) -> float:
    """
    Quantization Index Modulation (QIM) for embedding
    
    Args:
        coef: DCT coefficient value
        bit: Bit to embed (0 or 1)
        k: Modulation step size
        
    Returns:
        Modified coefficient
    """
    # Quantize
    quantized = round(coef / k)
    
    # Make even for bit 0, odd for bit 1
    if bit == 0:
        if quantized % 2 == 1:
            quantized += 1
    else:  # bit == 1
        if quantized % 2 == 0:
            quantized += 1
    
    # Dequantize
    return quantized * k

def extract_bit_from_coefficient(coef: float, k: int) -> int:
    """
    Extract bit from DCT coefficient using QIM extraction
    
    Args:
        coef: DCT coefficient value
        k: Modulation step size
        
    Returns:
        Extracted bit (0 or 1)
    """
    # Simple and robust: quantize and check parity
    # This matches the embedding logic where we make quantized values even/odd
    quantized = round(coef / k)
    return quantized % 2

def embed_in_block(block: np.ndarray, bits: List[int], k: int) -> np.ndarray:
    """
    Embed bits into one 8x8 DCT block
    
    Args:
        block: 8x8 image block
        bits: List of 8 bits to embed
        k: Embedding strength
        
    Returns:
        Modified block
    """
    # Convert to float
    block_float = block.astype(np.float32)
    
    # Apply DCT
    dct_block = cv2.dct(block_float)
    
    # Embed bits at specified positions
    for bit, (i, j) in zip(bits, EMBED_POSITIONS):
        dct_block[i, j] = quantization_index_modulation(dct_block[i, j], bit, k)
    
    # Apply IDCT
    modified_block = cv2.idct(dct_block)
    
    # Clip to valid range
    modified_block = np.clip(modified_block, 0, 255)
    
    return modified_block.astype(np.uint8)

def extract_from_block(block: np.ndarray, k: int) -> List[int]:
    """
    Extract bits from one 8x8 DCT block
    
    Args:
        block: 8x8 image block
        k: Embedding strength
        
    Returns:
        List of 8 extracted bits
    """
    # Convert to float
    block_float = block.astype(np.float32)
    
    # Apply DCT
    dct_block = cv2.dct(block_float)
    
    # Extract bits
    bits = []
    for i, j in EMBED_POSITIONS:
        bit = extract_bit_from_coefficient(dct_block[i, j], k)
        bits.append(bit)
    
    return bits

def embed_message(cover_image: np.ndarray, 
                 secret_data: bytes,
                 password: str,
                 k: int = 3) -> np.ndarray:
    """
    Embed secret data into cover image using DCT
    
    Args:
        cover_image: Cover image (BGR)
        secret_data: Data to embed (already encrypted and ECC-protected)
        password: Password for PRNG seed
        k: Embedding strength (higher = more robust, lower quality)
        
    Returns:
        Stego image (BGR)
    """
    print("\n[INFO] Starting DCT Embedding...")
    
    # Convert to YCbCr
    ycbcr_image = rgb_to_ycbcr(cover_image)
    y_channel = ycbcr_image[:, :, 0].copy()
    
    # Convert data to bits
    bits = bytes_to_bits(secret_data)
    total_bits = len(bits)
    
    # Calculate blocks needed (8 bits per block)
    blocks_needed = (total_bits + 7) // 8
    
    print(f"  Data size: {len(secret_data)} bytes = {total_bits} bits")
    print(f"  Blocks needed: {blocks_needed}")
    
    # Select blocks
    # Use ignore_variance=True for deterministic selection that works during extraction
    seed = derive_seed_from_password(password)
    selected_blocks = select_embedding_blocks(y_channel, blocks_needed, seed, ignore_variance=True)
    
    # Embed data
    bit_index = 0
    for block_num, (i, j) in enumerate(selected_blocks):
        # Extract block
        block = y_channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE].copy()
        
        # Get 8 bits for this block
        block_bits = bits[bit_index:bit_index+8]
        
        # Pad if necessary
        while len(block_bits) < 8:
            block_bits.append(0)
        
        # Embed
        modified_block = embed_in_block(block, block_bits, k)
        
        # Put back
        y_channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = modified_block
        
        bit_index += 8
        
        if (block_num + 1) % 100 == 0:
            print(f"  Embedded {block_num + 1}/{blocks_needed} blocks...")
    
    # Reconstruct image
    ycbcr_image[:, :, 0] = y_channel
    stego_image = ycbcr_to_rgb(ycbcr_image)
    
    print("[OK] Embedding complete!")
    
    return stego_image

def extract_message(stego_image: np.ndarray,
                   data_length: int,
                   password: str,
                   k: int = 3) -> bytes:
    """
    Extract secret data from stego image
    
    Args:
        stego_image: Stego image (BGR)
        data_length: Length of embedded data in bytes
        password: Password for PRNG seed
        k: Embedding strength used during embedding
        
    Returns:
        Extracted data (still encrypted and ECC-protected)
    """
    print("\n[INFO] Starting DCT Extraction...")
    
    # Convert to YCbCr
    ycbcr_image = rgb_to_ycbcr(stego_image)
    y_channel = ycbcr_image[:, :, 0]
    
    # Calculate blocks needed
    total_bits = data_length * 8
    blocks_needed = (total_bits + 7) // 8
    
    print(f"  Extracting {data_length} bytes = {total_bits} bits")
    print(f"  Blocks needed: {blocks_needed}")
    
    # Select same blocks as embedding
    # Use ignore_variance=True to ensure we get the same blocks even if variance changed
    seed = derive_seed_from_password(password)
    selected_blocks = select_embedding_blocks(y_channel, blocks_needed, seed, ignore_variance=True)
    
    # Extract data
    all_bits = []
    for block_num, (i, j) in enumerate(selected_blocks):
        # Extract block
        block = y_channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
        
        # Extract bits
        block_bits = extract_from_block(block, k)
        all_bits.extend(block_bits)
        
        if (block_num + 1) % 100 == 0:
            print(f"  Extracted {block_num + 1}/{blocks_needed} blocks...")
    
    # Trim to exact length
    all_bits = all_bits[:total_bits]
    
    # Convert to bytes
    extracted_data = bits_to_bytes(all_bits)
    
    # Ensure we have the exact length (pad if necessary)
    if len(extracted_data) < data_length:
        # Pad with zeros if needed
        extracted_data = extracted_data + b'\x00' * (data_length - len(extracted_data))
    elif len(extracted_data) > data_length:
        # Trim if too long
        extracted_data = extracted_data[:data_length]
    
    print(f"[OK] Extraction complete! Extracted {len(extracted_data)} bytes")
    
    return extracted_data

if __name__ == "__main__":
    # Test the module
    print("Testing DCT engine module...\n")
    
    # ⚙ CHANGE THIS - Use your test image path
    test_image_path = "dataset/cover_images/test.png"
    
    try:
        from src.preprocess import load_image
        
        # Load image
        cover = load_image(test_image_path)
        
        # Test data
        test_data = b"Hello, this is a secret message!" * 10  # ~320 bytes
        password = "TestPassword123"
        
        print(f"Test data: {len(test_data)} bytes")
        
        # Embed
        stego = embed_message(cover, test_data, password, k=3)
        
        # Extract
        extracted = extract_message(stego, len(test_data), password, k=3)
        
        # Verify
        print(f"\n[OK] Original length: {len(test_data)}")
        print(f"[OK] Extracted length: {len(extracted)}")
        print(f"[OK] Match: {test_data == extracted}")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

