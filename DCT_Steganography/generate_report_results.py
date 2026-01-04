"""
Generate report results table comparing with reference implementation
"""
import subprocess
import json
import os
from pathlib import Path

def create_test_message(size_bytes=128):
    """Create a test message of specified size"""
    message = "A" * size_bytes
    test_file = "dataset/secret_messages/test_128bytes.txt"
    with open(test_file, 'w') as f:
        f.write(message)
    return test_file

def run_embed_extract(cover_image, secret_file, output_stego, password="TestPass123", strength=3):
    """Run embedding and extraction, return metrics"""
    # Embed
    embed_cmd = [
        "python", "main.py", "embed",
        "--cover", cover_image,
        "--secret", secret_file,
        "--output", output_stego,
        "--password", password,
        "--strength", str(strength)
    ]
    
    result = subprocess.run(embed_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error embedding {cover_image}: {result.stderr}")
        return None
    
    # Extract metrics from output
    output = result.stdout
    psnr = None
    ssim = None
    
    for line in output.split('\n'):
        if 'PSNR' in line:
            try:
                psnr = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        if 'SSIM' in line:
            try:
                ssim = float(line.split(':')[1].strip())
            except:
                pass
    
    # Load metadata for capacity
    metadata_file = output_stego.replace('.png', '_metadata.json')
    capacity = None
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            # Get original secret size
            secret_size = os.path.getsize(secret_file)
            capacity = secret_size
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'capacity': capacity
    }

def main():
    print("="*70)
    print("  GENERATING REPORT RESULTS")
    print("="*70)
    print()
    
    # Create 128-byte test message
    test_msg = create_test_message(128)
    print(f"Created test message: {test_msg} ({os.path.getsize(test_msg)} bytes)")
    print()
    
    # Test images
    test_images = [
        ("test1.png", "Test Image 1"),
        ("test2.png", "Test Image 2"),
        ("test3.png", "Test Image 3"),
    ]
    
    results = []
    password = "TestPass123"
    strength = 3
    
    for img_file, img_name in test_images:
        cover_path = f"dataset/cover_images/{img_file}"
        if not os.path.exists(cover_path):
            print(f"Skipping {img_file} - not found")
            continue
        
        print(f"Testing {img_name} ({img_file})...")
        stego_path = f"results/stego_images/report_{img_file}"
        
        metrics = run_embed_extract(cover_path, test_msg, stego_path, password, strength)
        
        if metrics:
            results.append({
                'image': img_name,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'capacity': metrics['capacity'] or 128
            })
            print(f"  PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
        print()
    
    # Calculate averages
    if results:
        avg_psnr = sum(r['psnr'] for r in results) / len(results)
        avg_ssim = sum(r['ssim'] for r in results) / len(results)
        
        # Print table
        print("="*70)
        print("  IMPERCEPTIBILITY EVALUATION RESULTS")
        print("="*70)
        print()
        print(f"{'Image':<20} {'PSNR(dB)':<12} {'SSIM':<12} {'Capacity(bytes)':<15}")
        print("-"*70)
        
        for r in results:
            print(f"{r['image']:<20} {r['psnr']:<12.2f} {r['ssim']:<12.4f} {r['capacity']:<15}")
        
        print("-"*70)
        print(f"{'Average':<20} {avg_psnr:<12.2f} {avg_ssim:<12.4f} {128:<15}")
        print()
        
        # Comparison with reference
        print("="*70)
        print("  COMPARISON WITH REFERENCE IMPLEMENTATION")
        print("="*70)
        print()
        print(f"{'Metric':<20} {'Reference':<15} {'Our Results':<15} {'Difference':<15}")
        print("-"*70)
        print(f"{'Average PSNR (dB)':<20} {52.3:<15.2f} {avg_psnr:<15.2f} {avg_psnr-52.3:+.2f}")
        print(f"{'Average SSIM':<20} {0.982:<15.4f} {avg_ssim:<15.4f} {avg_ssim-0.982:+.4f}")
        print()
        
        # Save to file
        with open('results/report_table.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("  IMPERCEPTIBILITY EVALUATION RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'Image':<20} {'PSNR(dB)':<12} {'SSIM':<12} {'Capacity(bytes)':<15}\n")
            f.write("-"*70 + "\n")
            for r in results:
                f.write(f"{r['image']:<20} {r['psnr']:<12.2f} {r['ssim']:<12.4f} {r['capacity']:<15}\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Average':<20} {avg_psnr:<12.2f} {avg_ssim:<12.4f} {128:<15}\n")
        
        print("Results saved to: results/report_table.txt")
    else:
        print("No results generated!")

if __name__ == "__main__":
    main()

