"""
Module 4: Evaluation and Attack Simulation
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Dict
import io
from PIL import Image

def calculate_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio
    
    Args:
        original: Original image
        modified: Modified image
        
    Returns:
        PSNR in dB
    """
    return psnr(original, modified)

def calculate_ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index
    
    Args:
        original: Original image
        modified: Modified image
        
    Returns:
        SSIM value (0-1)
    """
    return ssim(original, modified, channel_axis=2)

def calculate_ber(original_bits: bytes, extracted_bits: bytes) -> float:
    """
    Calculate Bit Error Rate
    
    Args:
        original_bits: Original data
        extracted_bits: Extracted data
        
    Returns:
        BER as percentage
    """
    min_len = min(len(original_bits), len(extracted_bits))
    
    errors = 0
    total_bits = min_len * 8
    
    for i in range(min_len):
        xor = original_bits[i] ^ extracted_bits[i]
        errors += bin(xor).count('1')
    
    ber = (errors / total_bits) * 100 if total_bits > 0 else 100
    return ber

def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """
    Apply JPEG compression attack
    
    Args:
        image: Input image
        quality: JPEG quality (1-100)
        
    Returns:
        Compressed image
    """
    # Encode as JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    
    # Decode back
    compressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    
    return compressed

def apply_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian noise attack
    
    Args:
        image: Input image
        sigma: Standard deviation of noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def apply_salt_pepper_noise(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Apply salt and pepper noise
    
    Args:
        image: Input image
        amount: Proportion of pixels to corrupt
        
    Returns:
        Noisy image
    """
    noisy = image.copy()
    
    # Salt
    num_salt = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255
    
    # Pepper
    num_pepper = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0
    
    return noisy

def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filter
    
    Args:
        image: Input image
        kernel_size: Filter kernel size
        
    Returns:
        Filtered image
    """
    return cv2.medianBlur(image, kernel_size)

def apply_attack(image: np.ndarray, attack_type: str, **params) -> np.ndarray:
    """
    Apply specified attack to image
    
    Args:
        image: Input image
        attack_type: Type of attack
        **params: Attack parameters
        
    Returns:
        Attacked image
    """
    if attack_type == 'jpeg':
        return apply_jpeg_compression(image, params.get('quality', 85))
    elif attack_type == 'gaussian':
        return apply_gaussian_noise(image, params.get('sigma', 5))
    elif attack_type == 'salt_pepper':
        return apply_salt_pepper_noise(image, params.get('amount', 0.01))
    elif attack_type == 'median':
        return apply_median_filter(image, params.get('kernel', 3))
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

def evaluate_imperceptibility(cover: np.ndarray, stego: np.ndarray) -> Dict[str, float]:
    """
    Evaluate imperceptibility metrics
    
    Args:
        cover: Cover image
        stego: Stego image
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'psnr': calculate_psnr(cover, stego),
        'ssim': calculate_ssim(cover, stego)
    }
    
    return metrics

def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key.upper():20s}: {value:.4f}")
        else:
            print(f"  {key.upper():20s}: {value}")
    
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Test the module
    print("Testing evaluation module...\n")
    
    # ⚙ CHANGE THIS - Use your test image
    test_image_path = "dataset/cover_images/test.png"
    
    try:
        from src.preprocess import load_image
        
        # Load image
        image = load_image(test_image_path)
        
        # Test JPEG compression
        print("Testing JPEG compression attacks...")
        for quality in [95, 85, 75, 65]:
            compressed = apply_jpeg_compression(image, quality)
            metrics = evaluate_imperceptibility(image, compressed)
            print_metrics(metrics, f"JPEG Quality {quality}")
        
        # Test noise
        print("\nTesting Gaussian noise attacks...")
        for sigma in [2, 5, 10]:
            noisy = apply_gaussian_noise(image, sigma)
            metrics = evaluate_imperceptibility(image, noisy)
            print_metrics(metrics, f"Gaussian Noise σ={sigma}")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")

