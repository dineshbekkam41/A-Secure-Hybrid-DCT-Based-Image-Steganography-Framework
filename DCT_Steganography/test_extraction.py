"""
Test script to verify embedding/extraction works without image I/O
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_image, rgb_to_ycbcr, ycbcr_to_rgb
from src.security import prepare_data_for_embedding, recover_data_from_extraction
from src.dct_engine import embed_message, extract_message

# Load image
print("Loading image...")
cover = load_image("dataset/cover_images/test1.png")

# Test data
test_data = b"This is a test message for direct extraction test!"
password = "TestPassword123"

print(f"\nTest data: {len(test_data)} bytes")
print(f"Password: {password}")

# Prepare data
print("\nPreparing data...")
protected = prepare_data_for_embedding(test_data, password)
print(f"Protected data: {len(protected)} bytes")

# Embed
print("\nEmbedding...")
stego = embed_message(cover, protected, password, k=4)

# Extract WITHOUT saving/loading
print("\nExtracting (direct, no I/O)...")
extracted = extract_message(stego, len(protected), password, k=4)
print(f"Extracted data: {len(extracted)} bytes")

# Compare
print("\nComparing extracted vs original protected data...")
if extracted == protected:
    print("[SUCCESS] Extracted data matches protected data exactly!")
    
    # Try to recover
    print("\nAttempting recovery...")
    try:
        recovered = recover_data_from_extraction(extracted, password)
        if recovered == test_data:
            print("[SUCCESS] Recovered message matches original!")
        else:
            print("[ERROR] Recovered message doesn't match!")
            print(f"Original: {test_data}")
            print(f"Recovered: {recovered}")
    except Exception as e:
        print(f"[ERROR] Recovery failed: {e}")
else:
    print("[ERROR] Extracted data doesn't match protected data!")
    # Calculate bit errors
    errors = 0
    min_len = min(len(extracted), len(protected))
    for i in range(min_len):
        errors += bin(extracted[i] ^ protected[i]).count('1')
    total_bits = min_len * 8
    ber = (errors / total_bits) * 100 if total_bits > 0 else 100
    print(f"Bit errors: {errors} out of {total_bits} bits ({ber:.2f}%)")

