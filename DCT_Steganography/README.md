# DCT Steganography System

A complete implementation of DCT-based image steganography with AES encryption and Reed-Solomon error correction.

## Features

- **DCT-based Embedding**: Uses Discrete Cosine Transform for robust steganography
- **AES-256-GCM Encryption**: Secure encryption of secret messages
- **Reed-Solomon ECC**: Error correction for robustness against attacks
- **Quality Metrics**: PSNR and SSIM evaluation
- **Attack Simulation**: Test robustness against JPEG compression, noise, and filters

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create test images:
```bash
python download_test_images.py
```

## Usage

### Embedding

```bash
python main.py embed \
    --cover dataset/cover_images/test1.png \
    --secret dataset/secret_messages/message.txt \
    --output results/stego_images/stego1.png \
    --password MySecurePassword123 \
    --strength 3
```

### Extraction

```bash
python main.py extract \
    --stego results/stego_images/stego1.png \
    --output results/extracted_messages/recovered.txt \
    --password MySecurePassword123
```

### Testing Robustness

```bash
python main.py test \
    --cover dataset/cover_images/test1.png \
    --stego results/stego_images/stego1.png \
    --password MySecurePassword123 \
    --secret dataset/secret_messages/message.txt
```

## Project Structure

```
DCT_Steganography/
├── dataset/
│   ├── cover_images/      # Cover images
│   └── secret_messages/    # Secret messages
├── src/
│   ├── preprocess.py      # Image preprocessing
│   ├── security.py         # Encryption & ECC
│   ├── dct_engine.py       # DCT embedding/extraction
│   └── evaluation.py       # Metrics & attacks
├── results/
│   ├── stego_images/       # Stego images
│   ├── attacked_images/    # Attacked images
│   └── extracted_messages/ # Recovered messages
├── main.py                 # Main CLI interface
└── requirements.txt        # Dependencies
```

## Parameters

- `--strength`: Embedding strength (1-10, default: 3)
  - Higher = more robust but lower quality
  - Lower = higher quality but less robust

## Visualization

```bash
python visualize_results.py
```

## Report Generation

```bash
python generate_report_tables.py
```

## License

Academic project - for educational purposes only.

