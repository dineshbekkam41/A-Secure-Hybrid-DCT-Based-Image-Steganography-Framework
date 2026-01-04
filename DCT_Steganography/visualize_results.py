import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.preprocess import load_image
from src.evaluation import calculate_psnr, calculate_ssim

# ⚙ CHANGE THESE PATHS
cover_path = "dataset/cover_images/test1.png"
stego_path = "results/stego_images/stego1.png"

# Load images
cover = load_image(cover_path)
stego = load_image(stego_path)

# Calculate metrics
psnr_val = calculate_psnr(cover, stego)
ssim_val = calculate_ssim(cover, stego)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Cover image
axes[0].imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
axes[0].set_title('Cover Image', fontsize=14)
axes[0].axis('off')

# Stego image
axes[1].imshow(cv2.cvtColor(stego, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Stego Image\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}', fontsize=14)
axes[1].axis('off')

# Difference (amplified)
diff = np.abs(cover.astype(float) - stego.astype(float))
diff_amplified = diff * 10  # Amplify for visibility
diff_amplified = np.clip(diff_amplified, 0, 255).astype(np.uint8)

axes[2].imshow(cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2RGB))
axes[2].set_title('Difference (×10 amplified)', fontsize=14)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to results/comparison.png")
plt.show()

