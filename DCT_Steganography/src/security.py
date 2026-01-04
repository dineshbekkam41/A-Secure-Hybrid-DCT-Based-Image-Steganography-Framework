"""
Module 2: Security Layer (AES Encryption + Reed-Solomon ECC)
"""

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
from reedsolo import RSCodec
from typing import Tuple

# ⚙ CONFIGURATION
SALT = b'DCT_Stego_2024_Salt_Value_32bit'  # Fixed salt for key derivation
AES_KEY_SIZE = 32  # 256-bit AES
NONCE_SIZE = 12    # GCM nonce size
TAG_SIZE = 16      # GCM authentication tag size
RS_ECC_SYMBOLS = 64  # Reed-Solomon error correction symbols (can correct 32 byte errors)

def derive_key(password: str, salt: bytes = SALT) -> bytes:
    """
    Derive AES key from password using PBKDF2
    
    Args:
        password: User password
        salt: Salt for key derivation
        
    Returns:
        32-byte AES key
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(password.encode())
    return key

def aes_encrypt(data: bytes, password: str) -> Tuple[bytes, bytes, bytes]:
    """
    Encrypt data using AES-256-GCM
    
    Args:
        data: Plain data to encrypt
        password: Encryption password
        
    Returns:
        Tuple of (ciphertext, nonce, tag)
    """
    # Derive key from password
    key = derive_key(password)
    
    # Generate random nonce
    nonce = os.urandom(NONCE_SIZE)
    
    # Create cipher
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(nonce)
    )
    encryptor = cipher.encryptor()
    
    # Encrypt
    ciphertext = encryptor.update(data) + encryptor.finalize()
    tag = encryptor.tag
    
    print(f"[OK] Encrypted {len(data)} bytes -> {len(ciphertext)} bytes")
    
    return ciphertext, nonce, tag

def aes_decrypt(ciphertext: bytes, nonce: bytes, tag: bytes, password: str) -> bytes:
    """
    Decrypt data using AES-256-GCM
    
    Args:
        ciphertext: Encrypted data
        nonce: Nonce used during encryption
        tag: Authentication tag
        password: Decryption password
        
    Returns:
        Decrypted plaintext
    """
    # Derive key from password
    key = derive_key(password)
    
    # Create cipher
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(nonce, tag)
    )
    decryptor = cipher.decryptor()
    
    # Decrypt
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    
    print(f"[OK] Decrypted {len(ciphertext)} bytes -> {len(plaintext)} bytes")
    
    return plaintext

def apply_reed_solomon(data: bytes, nsym: int = RS_ECC_SYMBOLS) -> bytes:
    """
    Apply Reed-Solomon error correction coding
    
    Args:
        data: Input data
        nsym: Number of error correction symbols (can correct nsym/2 errors)
        
    Returns:
        Data with ECC
    """
    rs = RSCodec(nsym)
    encoded = rs.encode(data)
    
    overhead = len(encoded) - len(data)
    print(f"[OK] Applied Reed-Solomon ECC: {len(data)} -> {len(encoded)} bytes (+{overhead} bytes)")
    
    return encoded

def decode_reed_solomon(data: bytes, nsym: int = RS_ECC_SYMBOLS) -> bytes:
    """
    Decode and correct errors using Reed-Solomon
    
    Args:
        data: Encoded data (possibly corrupted)
        nsym: Number of error correction symbols used during encoding
        
    Returns:
        Corrected data
    """
    rs = RSCodec(nsym)
    
    try:
        decoded_tuple = rs.decode(data)
        # rs.decode returns tuple (message, ecc) or just message depending on version
        if isinstance(decoded_tuple, tuple):
            decoded = decoded_tuple[0]
        else:
            decoded = decoded_tuple
        print(f"[OK] Reed-Solomon decode successful: {len(data)} -> {len(decoded)} bytes")
        return decoded
    except Exception as e:
        print(f"[ERROR] Reed-Solomon decode failed: {e}")
        raise ValueError("Data corruption beyond repair")

def prepare_data_for_embedding(secret_data: bytes, password: str) -> bytes:
    """
    Complete security pipeline: Encrypt + ECC
    
    Args:
        secret_data: Raw secret message
        password: Encryption password
        
    Returns:
        Protected data ready for embedding
    """
    # Encrypt
    ciphertext, nonce, tag = aes_encrypt(secret_data, password)
    
    # Combine components
    combined = ciphertext + nonce + tag
    
    # Apply ECC
    protected = apply_reed_solomon(combined, RS_ECC_SYMBOLS)
    
    return protected

def recover_data_from_extraction(extracted_data: bytes, password: str) -> bytes:
    """
    Complete security pipeline: ECC decode + Decrypt
    
    Args:
        extracted_data: Data extracted from stego image
        password: Decryption password
        
    Returns:
        Original secret message
    """
    # Decode ECC
    decoded = decode_reed_solomon(extracted_data, RS_ECC_SYMBOLS)
    
    # Split components
    # decoded should be: ciphertext (variable) + nonce (12 bytes) + tag (16 bytes)
    if len(decoded) < 28:
        raise ValueError(f"Decoded data too short: {len(decoded)} bytes (need at least 28)")
    
    ciphertext = decoded[:-28]  # Everything except last 28 bytes
    nonce = decoded[-28:-16]    # 12 bytes nonce
    tag = decoded[-16:]          # 16 bytes tag
    
    # Ensure all are bytes
    if not isinstance(nonce, bytes):
        nonce = bytes(nonce)
    if not isinstance(tag, bytes):
        tag = bytes(tag)
    
    # Decrypt
    plaintext = aes_decrypt(ciphertext, nonce, tag, password)
    
    return plaintext

if __name__ == "__main__":
    # Test the module
    print("Testing security module...\n")
    
    # ⚙ CHANGE THIS - Test with your own message and password
    test_message = b"This is a top secret message for testing!"
    test_password = "MySecurePassword123"
    
    print(f"Original message: {test_message}")
    print(f"Original size: {len(test_message)} bytes\n")
    
    # Encrypt and apply ECC
    protected = prepare_data_for_embedding(test_message, test_password)
    print(f"\nProtected data size: {len(protected)} bytes")
    print(f"Overhead: {len(protected) - len(test_message)} bytes\n")
    
    # Simulate some corruption (flip 5 random bits)
    import random
    corrupted = bytearray(protected)
    for _ in range(5):
        pos = random.randint(0, len(corrupted) - 1)
        bit = random.randint(0, 7)
        corrupted[pos] ^= (1 << bit)
    corrupted = bytes(corrupted)
    print(f"Simulated {5} bit errors\n")
    
    # Recover
    try:
        recovered = recover_data_from_extraction(corrupted, test_password)
        print(f"\n[SUCCESS] Recovered message: {recovered}")
        print(f"Match: {recovered == test_message}")
    except Exception as e:
        print(f"[ERROR] Recovery failed: {e}")

