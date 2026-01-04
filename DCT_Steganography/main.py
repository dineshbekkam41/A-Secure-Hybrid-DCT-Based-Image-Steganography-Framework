#!/usr/bin/env python3
"""
DCT Steganography System - Main Interface
Academic Mini-Project
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_image, save_image, calculate_capacity
from src.security import prepare_data_for_embedding, recover_data_from_extraction
from src.dct_engine import embed_message, extract_message
from src.evaluation import (
    evaluate_imperceptibility, print_metrics, apply_attack,
    calculate_ber
)

def embed_workflow(args):
    """
    Complete embedding workflow
    """
    print("\n" + "="*60)
    print("  [EMBED] DCT STEGANOGRAPHY - EMBEDDING MODE")
    print("="*60)
    
    try:
        # 1. Load cover image
        print("\n[1/6] Loading cover image...")
        cover_image = load_image(args.cover)
        
        # Check capacity
        capacity = calculate_capacity(cover_image.shape, bits_per_block=8)
        print(f"  Image capacity: ~{capacity['capacity_bytes']} bytes")
        
        # 2. Load secret message
        print("\n[2/6] Loading secret message...")
        with open(args.secret, 'rb') as f:
            secret_data = f.read()
        
        print(f"  Secret data size: {len(secret_data)} bytes")
        
        # 3. Apply security (Encryption + ECC)
        print("\n[3/6] Applying security layer...")
        protected_data = prepare_data_for_embedding(secret_data, args.password)
        
        print(f"  Protected data size: {len(protected_data)} bytes")
        
        # Check if it fits
        if len(protected_data) > capacity['capacity_bytes']:
            print(f"\n[ERROR] Data too large!")
            print(f"  Need: {len(protected_data)} bytes")
            print(f"  Available: {capacity['capacity_bytes']} bytes")
            return
        
        # 4. Embed using DCT
        print("\n[4/6] Embedding data into cover image...")
        stego_image = embed_message(
            cover_image,
            protected_data,
            args.password,
            k=args.strength
        )
        
        # 5. Save stego image
        print("\n[5/6] Saving stego image...")
        save_image(args.output, stego_image)
        
        # 6. Calculate metrics
        print("\n[6/6] Calculating quality metrics...")
        metrics = evaluate_imperceptibility(cover_image, stego_image)
        
        # Display results
        print("\n" + "="*60)
        print("  [SUCCESS] EMBEDDING COMPLETE!")
        print("="*60)
        print(f"  Cover Image      : {args.cover}")
        print(f"  Secret Message   : {args.secret} ({len(secret_data)} bytes)")
        print(f"  Stego Image      : {args.output}")
        print(f"  Password         : {'*' * len(args.password)}")
        print(f"  Embedding Strength: {args.strength}")
        print(f"  PSNR             : {metrics['psnr']:.2f} dB")
        print(f"  SSIM             : {metrics['ssim']:.4f}")
        print("="*60)
        
        # Save metadata for extraction
        metadata = {
            'data_length': len(protected_data),
            'strength': args.strength,
            'secret_filename': os.path.basename(args.secret)
        }
        
        metadata_path = args.output.replace('.png', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[SAVED] Metadata saved to: {metadata_path}")
        print("    (Keep this file for extraction!)\n")
        
    except Exception as e:
        print(f"\n[ERROR] Error during embedding: {e}")
        import traceback
        traceback.print_exc()

def extract_workflow(args):
    """
    Complete extraction workflow
    """
    print("\n" + "="*60)
    print("  [EXTRACT] DCT STEGANOGRAPHY - EXTRACTION MODE")
    print("="*60)
    
    try:
        # Load metadata if available
        metadata_path = args.stego.replace('.png', '_metadata.json')
        if os.path.exists(metadata_path) and not args.length:
            print(f"\n[INFO] Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                args.length = metadata['data_length']
                args.strength = metadata.get('strength', args.strength)
        
        if not args.length:
            print("\n[ERROR] Data length required!")
            print("   Use --length parameter or ensure metadata file exists")
            return
        
        # 1. Load stego image
        print("\n[1/4] Loading stego image...")
        stego_image = load_image(args.stego)
        
        # 2. Extract data
        print("\n[2/4] Extracting data from stego image...")
        extracted_data = extract_message(
            stego_image,
            args.length,
            args.password,
            k=args.strength
        )
        
        # 3. Recover (ECC + Decryption)
        print("\n[3/4] Recovering secret message...")
        print(f"  Extracted data length: {len(extracted_data)} bytes (expected: {args.length} bytes)")
        
        # Try to load original protected data for comparison (if available)
        try:
            secret_data = recover_data_from_extraction(extracted_data, args.password)
        except Exception as e:
            print(f"\n[ERROR] Recovery failed: {e}")
            print("   Possible reasons:")
            print("   - Wrong password")
            print("   - Image was modified too much")
            print("   - Wrong data length")
            print("   - Too many bit errors from DCT/IDCT rounding")
            print("\n   Tip: Try re-embedding with a higher --strength value (e.g., 4 or 5)")
            return
        
        # 4. Save recovered message
        print("\n[4/4] Saving recovered message...")
        with open(args.output, 'wb') as f:
            f.write(secret_data)
        
        print("\n" + "="*60)
        print("  [SUCCESS] EXTRACTION COMPLETE!")
        print("="*60)
        print(f"  Stego Image      : {args.stego}")
        print(f"  Recovered Message: {args.output}")
        print(f"  Message Size     : {len(secret_data)} bytes")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Error during extraction: {e}")
        import traceback
        traceback.print_exc()

def test_workflow(args):
    """
    Complete test workflow with attacks
    """
    print("\n" + "="*60)
    print("  [TEST] DCT STEGANOGRAPHY - TEST MODE")
    print("="*60)
    
    try:
        # Load images
        print("\n[1/5] Loading images...")
        cover_image = load_image(args.cover)
        stego_image = load_image(args.stego)
        
        # Load metadata
        metadata_path = args.stego.replace('.png', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                data_length = metadata['data_length']
                strength = metadata.get('strength', 3)
        else:
            print("[ERROR] Metadata file not found!")
            return
        
        # Original extraction (no attack)
        print("\n[2/5] Testing extraction without attack...")
        extracted_clean = extract_message(stego_image, data_length, args.password, k=strength)
        secret_clean = recover_data_from_extraction(extracted_clean, args.password)
        
        # Load original secret for BER calculation
        secret_file = metadata.get('secret_filename', args.secret)
        if args.secret and os.path.exists(args.secret):
            with open(args.secret, 'rb') as f:
                original_secret = f.read()
        else:
            original_secret = secret_clean
        
        # Imperceptibility metrics
        print("\n[3/5] Calculating imperceptibility metrics...")
        imperceptibility = evaluate_imperceptibility(cover_image, stego_image)
        print_metrics(imperceptibility, "IMPERCEPTIBILITY")
        
        # Define attacks
        attacks = {
            'JPEG Q=95': {'type': 'jpeg', 'quality': 95},
            'JPEG Q=85': {'type': 'jpeg', 'quality': 85},
            'JPEG Q=75': {'type': 'jpeg', 'quality': 75},
            'JPEG Q=65': {'type': 'jpeg', 'quality': 65},
            'Gaussian σ=2': {'type': 'gaussian', 'sigma': 2},
            'Gaussian σ=5': {'type': 'gaussian', 'sigma': 5},
            'Salt&Pepper 0.005': {'type': 'salt_pepper', 'amount': 0.005},
            'Median 3x3': {'type': 'median', 'kernel': 3}
        }
        
        # Test robustness
        print("\n[4/5] Testing robustness against attacks...")
        results = {}
        
        for attack_name, attack_params in attacks.items():
            print(f"\n  Testing: {attack_name}")
            
            try:
                # Apply attack
                attacked_image = apply_attack(stego_image, **attack_params)
                
                # Save attacked image
                attacked_path = f"results/attacked_images/{attack_name.replace(' ', '_').replace('=', '_')}.png"
                os.makedirs(os.path.dirname(attacked_path), exist_ok=True)
                save_image(attacked_path, attacked_image)
                
                # Try extraction
                extracted_attacked = extract_message(attacked_image, data_length, args.password, k=strength)
                
                try:
                    secret_attacked = recover_data_from_extraction(extracted_attacked, args.password)
                    ber = calculate_ber(original_secret, secret_attacked)
                    success = True
                    print(f"    [OK] BER: {ber:.2f}%")
                except:
                    ber = calculate_ber(original_secret, extracted_attacked[:len(original_secret)])
                    success = False
                    print(f"    [FAIL] Recovery failed, BER: {ber:.2f}%")
                
                results[attack_name] = {
                    'success': success,
                    'ber': ber
                }
                
            except Exception as e:
                print(f"    [FAIL] Failed: {e}")
                results[attack_name] = {
                    'success': False,
                    'ber': 100.0
                }
        
        # Display summary
        print("\n[5/5] Generating summary...")
        print("\n" + "="*60)
        print("  [RESULTS] ROBUSTNESS TEST RESULTS")
        print("="*60)
        print(f"  {'Attack':<25} {'Success':<10} {'BER (%)':<10}")
        print("-"*60)
        
        for attack_name, result in results.items():
            success_icon = "[OK]" if result['success'] else "[FAIL]"
            print(f"  {attack_name:<25} {success_icon:<10} {result['ber']:<10.2f}")
        
        print("="*60)
        
        # Save results
        results_file = "results/test_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        full_results = {
            'imperceptibility': imperceptibility,
            'robustness': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\n[SAVED] Full results saved to: {results_file}\n")
        
    except Exception as e:
        print(f"\n[ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DCT Steganography System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Embed secret message
  python main.py embed --cover dataset/cover_images/lena.png \\
                       --secret dataset/secret_messages/message.txt \\
                       --output results/stego_images/stego.png \\
                       --password MyPassword123

  # Extract secret message
  python main.py extract --stego results/stego_images/stego.png \\
                         --output results/extracted_messages/recovered.txt \\
                         --password MyPassword123

  # Test robustness
  python main.py test --cover dataset/cover_images/lena.png \\
                      --stego results/stego_images/stego.png \\
                      --password MyPassword123 \\
                      --secret dataset/secret_messages/message.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Embed secret message into cover image')
    embed_parser.add_argument('--cover', required=True, help='Path to cover image')
    embed_parser.add_argument('--secret', required=True, help='Path to secret message file')
    embed_parser.add_argument('--output', required=True, help='Path to output stego image')
    embed_parser.add_argument('--password', required=True, help='Password for encryption')
    embed_parser.add_argument('--strength', type=int, default=3, help='Embedding strength (default: 3)')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract secret message from stego image')
    extract_parser.add_argument('--stego', required=True, help='Path to stego image')
    extract_parser.add_argument('--output', required=True, help='Path to output message file')
    extract_parser.add_argument('--password', required=True, help='Password for decryption')
    extract_parser.add_argument('--length', type=int, help='Data length in bytes (auto-detected from metadata if available)')
    extract_parser.add_argument('--strength', type=int, default=3, help='Embedding strength (default: 3)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test robustness against attacks')
    test_parser.add_argument('--cover', required=True, help='Path to cover image')
    test_parser.add_argument('--stego', required=True, help='Path to stego image')
    test_parser.add_argument('--password', required=True, help='Password for decryption')
    test_parser.add_argument('--secret', help='Path to original secret file (for BER calculation)')
    
    args = parser.parse_args()
    
    if args.command == 'embed':
        embed_workflow(args)
    elif args.command == 'extract':
        extract_workflow(args)
    elif args.command == 'test':
        test_workflow(args)
    else:
        parser.print_help()

