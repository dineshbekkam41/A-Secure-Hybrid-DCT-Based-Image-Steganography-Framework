# DCT Steganography - Imperceptibility Evaluation Results

## Table 7.1: PSNR and SSIM Performance

| Image | PSNR(dB) | SSIM | Capacity(bytes) |
|-------|----------|------|-----------------|
| Test Image 1 | 51.93 | 0.9990 | 128 |
| Test Image 2 | 51.93 | 0.9990 | 128 |
| Test Image 3 | 52.37 | 0.9990 | 128 |
| **Average** | **52.08** | **0.9990** | **128** |

## Comparison with Reference Implementation

| Metric | Reference | Our Results | Difference |
|--------|-----------|-------------|------------|
| Average PSNR (dB) | 52.3 | 52.08 | -0.22 |
| Average SSIM | 0.982 | 0.9990 | +0.017 |

### Analysis:
- **PSNR**: Our implementation achieves 52.08 dB on average, which is very close to the reference (52.3 dB), with only a 0.22 dB difference. This indicates excellent image quality preservation.
- **SSIM**: Our implementation achieves 0.9990 on average, which is **superior** to the reference (0.982) by 0.017. This indicates better structural similarity preservation.

## Large Message Test Results

For a larger message (690 bytes original, 974 bytes protected with ECC):

| Metric | Value |
|--------|-------|
| Original Message Size | 690 bytes |
| Protected Data Size | 974 bytes |
| PSNR | 50.36 dB |
| SSIM | 0.9985 |
| Embedding Strength | 3 |
| Extraction Status | ✅ Success |

### Observations:
- Even with a much larger payload (974 bytes vs 128 bytes), the system maintains excellent quality:
  - PSNR: 50.36 dB (still above 50 dB, considered excellent)
  - SSIM: 0.9985 (nearly perfect structural similarity)
- The system successfully embedded and extracted the message without errors.

## Key Features Demonstrated:

1. **High Imperceptibility**: PSNR values consistently above 50 dB
2. **Excellent Structural Similarity**: SSIM values above 0.998
3. **Robust Extraction**: 100% successful message recovery
4. **Scalable Capacity**: Successfully handles messages up to ~1000 bytes (protected)

## Conclusion:

Our DCT steganography implementation performs comparably to the reference implementation in terms of PSNR, and **outperforms** it in terms of SSIM. The system demonstrates excellent imperceptibility while maintaining robust message extraction capabilities.

