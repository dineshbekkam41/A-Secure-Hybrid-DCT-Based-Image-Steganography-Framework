# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Create Test Images (if not already created)

```bash
python download_test_images.py
```

This creates test images in `dataset/cover_images/`:
- test1.png (512x512)
- test2.png (512x512)
- test3.png (1024x1024)

## Step 3: Test Embedding

```bash
python main.py embed --cover dataset/cover_images/test1.png --secret dataset/secret_messages/message.txt --output results/stego_images/stego1.png --password MyPassword123 --strength 3
```

## Step 4: Test Extraction

```bash
python main.py extract --stego results/stego_images/stego1.png --output results/extracted_messages/recovered.txt --password MyPassword123
```

## Step 5: Verify

Compare the original and recovered messages:

```bash
# On Windows PowerShell
Compare-Object (Get-Content dataset/secret_messages/message.txt) (Get-Content results/extracted_messages/recovered.txt)

# On Linux/Mac
diff dataset/secret_messages/message.txt results/extracted_messages/recovered.txt
```

## Step 6: Test Robustness

```bash
python main.py test --cover dataset/cover_images/test1.png --stego results/stego_images/stego1.png --password MyPassword123 --secret dataset/secret_messages/message.txt
```

## Step 7: Visualize Results

```bash
python visualize_results.py
```

(Remember to update the paths in visualize_results.py first!)

## Step 8: Generate Report

```bash
python generate_report_tables.py
```

## Troubleshooting

### "Could not load image"
- Make sure the image path is correct
- Check that the image file exists
- Verify the image is a valid image format (PNG, JPG, etc.)

### "Not enough suitable blocks"
- Use a larger image (at least 512x512 recommended)
- Reduce the size of your secret message
- Try adjusting MIN_VARIANCE and MAX_VARIANCE in src/preprocess.py

### "Data corruption beyond repair"
- Check that you're using the correct password
- Verify the image hasn't been heavily modified
- Ensure the data length is correct

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're running from the DCT_Steganography directory
- Check Python version (3.7+ required)

## Parameters Explained

- `--strength`: Embedding strength (1-10)
  - 1-2: High quality, less robust
  - 3-4: Balanced (recommended)
  - 5-10: Lower quality, more robust

- `--password`: Used for:
  - AES encryption/decryption
  - PRNG seed generation for block selection

## File Structure

```
DCT_Steganography/
├── dataset/
│   ├── cover_images/      # Your cover images go here
│   └── secret_messages/    # Your secret messages go here
├── src/                    # Source code modules
├── results/                # Output files
│   ├── stego_images/       # Stego images
│   ├── attacked_images/    # Attacked images (from tests)
│   └── extracted_messages/ # Recovered messages
└── main.py                # Main entry point
```

## Next Steps

1. Try with your own images and messages
2. Experiment with different embedding strengths
3. Test with various attack scenarios
4. Analyze the results for your report

Good luck! 🚀

