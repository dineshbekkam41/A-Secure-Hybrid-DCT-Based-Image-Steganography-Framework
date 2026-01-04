# Quick Reference Guide - DCT Steganography Project

## 🎯 Project Summary in One Sentence
A Python-based image steganography system that hides secret messages in images using DCT (Discrete Cosine Transform) with AES-256-GCM encryption and Reed-Solomon error correction.

---

## 📚 Technologies Used & Where

| Technology | Purpose | Used In Files |
|------------|---------|---------------|
| **Python 3.7+** | Programming language | All files |
| **OpenCV** | Image processing, DCT operations | `preprocess.py`, `dct_engine.py`, `evaluation.py` |
| **NumPy** | Array operations, math | All source files |
| **Cryptography** | AES-256-GCM encryption | `security.py` |
| **Reed-Solomon** | Error correction | `security.py` |
| **Scikit-image** | PSNR, SSIM metrics | `evaluation.py` |

---

## 📁 File Structure & Purpose

```
DCT_Steganography/
├── src/
│   ├── preprocess.py      → Image loading, block selection
│   ├── security.py         → Encryption, error correction
│   ├── dct_engine.py       → DCT embedding/extraction
│   └── evaluation.py       → Quality metrics, attacks
├── main.py                 → CLI interface, workflow
├── requirements.txt        → Dependencies
└── [documentation files]
```

---

## 🔑 Key Algorithms

### 1. DCT (Discrete Cosine Transform)
- **What**: Converts image blocks to frequency domain
- **Where**: `src/dct_engine.py`, lines ~131, ~138
- **Library**: OpenCV (`cv2.dct()`, `cv2.idct()`)
- **Why**: Robust, compatible with JPEG

### 2. QIM (Quantization Index Modulation)
- **What**: Embeds bits by making coefficients even/odd
- **Where**: `src/dct_engine.py`, lines ~75-115
- **Library**: NumPy (rounding, modulo)
- **Why**: Simple, robust, reversible

### 3. AES-256-GCM
- **What**: Encryption with authentication
- **Where**: `src/security.py`, lines ~41-99
- **Library**: Cryptography library
- **Why**: Industry standard, secure

### 4. PBKDF2
- **What**: Key derivation from password
- **Where**: `src/security.py`, lines ~19-38
- **Library**: Cryptography library
- **Why**: Resistant to brute-force

### 5. Reed-Solomon
- **What**: Error correction coding
- **Where**: `src/security.py`, lines ~101-144
- **Library**: reedsolo
- **Why**: Corrects multiple byte errors

---

## 🔄 Workflow Steps

### Embedding:
1. Load cover image (OpenCV)
2. Load secret message (file I/O)
3. Encrypt message (Cryptography library)
4. Add error correction (Reed-Solomon)
5. Convert to YCbCr (OpenCV)
6. Select blocks (NumPy PRNG)
7. Embed in DCT domain (OpenCV DCT)
8. Save stego image (OpenCV)

### Extraction:
1. Load stego image (OpenCV)
2. Convert to YCbCr (OpenCV)
3. Select same blocks (NumPy PRNG)
4. Extract from DCT (OpenCV DCT)
5. Correct errors (Reed-Solomon)
6. Decrypt (Cryptography library)
7. Save recovered message

---

## 💻 Code Locations

### Image Loading
- **File**: `src/preprocess.py`
- **Function**: `load_image()`
- **Line**: ~24
- **Library**: OpenCV

### Block Selection
- **File**: `src/preprocess.py`
- **Function**: `select_embedding_blocks()`
- **Line**: ~91
- **Library**: NumPy

### Encryption
- **File**: `src/security.py`
- **Function**: `aes_encrypt()`
- **Line**: ~41
- **Library**: Cryptography

### DCT Embedding
- **File**: `src/dct_engine.py`
- **Function**: `embed_in_block()`
- **Line**: ~117
- **Library**: OpenCV

### Quality Metrics
- **File**: `src/evaluation.py`
- **Function**: `calculate_psnr()`, `calculate_ssim()`
- **Line**: ~13, ~26
- **Library**: Scikit-image

---

## 📊 Results Summary

- **PSNR**: 52.08 dB average (excellent)
- **SSIM**: 0.9990 average (nearly perfect)
- **Capacity**: Tested up to 690 bytes
- **Extraction Rate**: 100% success

---

## 🎓 For Presentation

### Key Points to Mention:
1. **Why DCT?** - Robust, compatible with JPEG, less perceptible
2. **Why AES-256-GCM?** - Industry standard, authenticated encryption
3. **Why Reed-Solomon?** - Corrects errors from DCT rounding
4. **Why YCbCr?** - Embed in luminance channel, less visible
5. **Why QIM?** - Simple, robust, reversible

### Technologies to Highlight:
- OpenCV for image processing
- Cryptography library for security
- NumPy for efficiency
- Python for ease of development

---

## 📖 Documentation Files

1. **PROJECT_REPORT.md** - Complete technical report
2. **PRESENTATION_GUIDE.md** - Oral presentation guide
3. **CODE_MAPPING.md** - Technology usage mapping
4. **QUICK_REFERENCE.md** - This file
5. **REPORT_RESULTS.md** - Results analysis

---

## 🚀 Quick Commands

```bash
# Embed
python main.py embed --cover image.png --secret message.txt \
  --output stego.png --password MyPass123

# Extract
python main.py extract --stego stego.png --output recovered.txt \
  --password MyPass123

# Test
python main.py test --cover image.png --stego stego.png \
  --password MyPass123
```

---

## ❓ Common Questions

**Q: Why Python?**
A: Rich libraries, easy development, scientific computing support

**Q: Why DCT over LSB?**
A: More robust, less perceptible, compatible with JPEG

**Q: Why AES-256-GCM?**
A: Industry standard, authenticated encryption, very secure

**Q: Why Reed-Solomon?**
A: Can correct multiple errors, standard in digital communications

**Q: What's the capacity?**
A: ~2457 bytes theoretical, tested up to 690 bytes successfully

---

**Use this as a quick reference during your presentation!**

