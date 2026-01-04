import json
import pandas as pd

# Load test results
with open('results/test_results.json', 'r') as f:
    results = json.load(f)

# Imperceptibility table
imperceptibility = results['imperceptibility']
print("\n" + "="*50)
print("IMPERCEPTIBILITY METRICS")
print("="*50)
print(f"PSNR: {imperceptibility['psnr']:.2f} dB")
print(f"SSIM: {imperceptibility['ssim']:.4f}")

# Robustness table
robustness = results['robustness']

data = []
for attack, metrics in robustness.items():
    data.append({
        'Attack': attack,
        'Success': '✓' if metrics['success'] else '✗',
        'BER (%)': f"{metrics['ber']:.2f}"
    })

df = pd.DataFrame(data)

print("\n" + "="*50)
print("ROBUSTNESS TEST RESULTS")
print("="*50)
print(df.to_string(index=False))

# Save as CSV for report
df.to_csv('results/robustness_results.csv', index=False)
print("\n✓ Results saved to results/robustness_results.csv")

