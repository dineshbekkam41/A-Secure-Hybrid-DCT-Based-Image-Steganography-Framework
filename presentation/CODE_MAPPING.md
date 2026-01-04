# Code Mapping: Technologies Used in Each File

This document shows exactly where each technology/library is used in the codebase.

## File-by-File Technology Usage

### 1. `src/preprocess.py`

#### OpenCV Usage:
```python
import cv2

# Line ~25: Load image
image = cv2.imread(image_path)  # Loads BGR image

# Line ~32: Convert BGR to YCbCr
ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Line ~39: Convert YCbCr back to BGR
bgr = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)

# Line ~172: Save image
cv2.imwrite(image_path, image)
```

**Why OpenCV here?**
- Industry-standard image I/O
- Efficient color space conversion
- Optimized C++ backend

#### NumPy Usage:
```python
import numpy as np

# Line ~60: Calculate variance
return np.var(block)  # Statistical variance calculation

# Line ~107: Array operations
height, width = y_channel.shape  # Get image dimensions

# Line ~130: Random selection
rng = np.random.RandomState(seed)  # Deterministic PRNG
selected_indices = rng.choice(len(suitable_blocks), num_blocks_needed, replace=False)
```

**Why NumPy here?**
- Efficient array operations
- Statistical functions (variance)
- Deterministic random number generation

---

### 2. `src/security.py`

#### Cryptography Library Usage:
```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Line ~31: Key derivation
kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),  # SHA-256 hash function
    length=32,                  # 256-bit key
    salt=SALT,
    iterations=100000           # Slow key derivation
)
key = kdf.derive(password.encode())

# Line ~57: AES encryption
cipher = Cipher(
    algorithms.AES(key),        # AES-256 algorithm
    modes.GCM(nonce)            # GCM mode (authenticated encryption)
)
encryptor = cipher.encryptor()
ciphertext = encryptor.update(data) + encryptor.finalize()
tag = encryptor.tag             # Authentication tag

# Line ~88: AES decryption
cipher = Cipher(
    algorithms.AES(key),
    modes.GCM(nonce, tag)       # Include tag for verification
)
decryptor = cipher.decryptor()
plaintext = decryptor.update(ciphertext) + decryptor.finalize()
```

**Why Cryptography library?**
- Industry-standard implementations
- Secure by default
- Well-maintained and audited
- Supports modern algorithms (AES-GCM)

#### Reed-Solomon Library Usage:
```python
from reedsolo import RSCodec

# Line ~112: Apply error correction
rs = RSCodec(64)  # 64 error correction symbols
encoded = rs.encode(data)  # Add redundancy

# Line ~131: Decode and correct errors
rs = RSCodec(64)
decoded = rs.decode(data)  # Corrects errors automatically
```

**Why Reed-Solomon?**
- Can correct multiple byte errors
- Efficient for burst errors
- Standard in digital communications

#### OS Library Usage:
```python
import os

# Line ~55: Generate random nonce
nonce = os.urandom(12)  # Cryptographically secure random bytes
```

**Why os.urandom?**
- Cryptographically secure
- System-level randomness
- Required for secure nonce generation

---

### 3. `src/dct_engine.py`

#### OpenCV DCT Usage:
```python
import cv2

# Line ~131: Forward DCT (spatial → frequency)
dct_block = cv2.dct(block_float)  # Convert 8x8 block to frequency domain

# Line ~138: Inverse DCT (frequency → spatial)
modified_block = cv2.idct(dct_block)  # Convert back to spatial domain

# Line ~160: DCT for extraction
dct_block = cv2.dct(block_float)  # Extract from frequency domain
```

**Why OpenCV DCT?**
- Optimized implementation
- Direct DCT/IDCT functions
- Efficient C++ backend
- Standard 8x8 block DCT

#### NumPy Usage:
```python
import numpy as np

# Line ~128: Type conversion
block_float = block.astype(np.float32)  # DCT requires float

# Line ~141: Clipping values
modified_block = np.clip(modified_block, 0, 255)  # Ensure valid pixel range

# Line ~143: Type conversion
return modified_block.astype(np.uint8)  # Convert back to integer

# Line ~48: Bit manipulation
bits.append((byte >> (7 - i)) & 1)  # Extract bits from bytes

# Line ~70: Bit to byte conversion
byte_val = (byte_val << 1) | bits[i + j]  # Reconstruct bytes from bits
```

**Why NumPy here?**
- Efficient array operations
- Type conversions
- Mathematical operations (rounding, clipping)
- Bit manipulation support

#### Hashlib Usage:
```python
import hashlib

# Line ~28: Derive seed from password
hash_obj = hashlib.sha256(password.encode())
seed = int.from_bytes(hash_obj.digest()[:4], 'big')
```

**Why Hashlib?**
- Standard library (no dependencies)
- SHA-256 for deterministic seed
- Cryptographic hash function

---

### 4. `src/evaluation.py`

#### Scikit-image Usage:
```python
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Line ~15: Calculate PSNR
return psnr(original, modified)  # Peak Signal-to-Noise Ratio

# Line ~37: Calculate SSIM
return ssim(original, modified, channel_axis=2)  # Structural Similarity Index
```

**Why Scikit-image?**
- Standard implementations
- Well-tested algorithms
- Accurate metrics
- Industry-standard

#### OpenCV Usage:
```python
import cv2

# Line ~62: JPEG compression attack
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
_, encimg = cv2.imencode('.jpg', image, encode_param)
compressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

# Line ~95: Median filter attack
return cv2.medianBlur(image, kernel_size)
```

**Why OpenCV here?**
- Image processing operations
- JPEG encoding/decoding
- Filter operations

#### NumPy Usage:
```python
import numpy as np

# Line ~75: Gaussian noise
noise = np.random.normal(0, sigma, image.shape)
noisy_image = image + noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Line ~88: Salt & pepper noise
coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
```

**Why NumPy?**
- Random number generation
- Array operations
- Mathematical operations

---

### 5. `main.py`

#### Standard Library Usage:
```python
import argparse  # Command-line argument parsing
import os        # File operations
import sys       # System operations
import json      # JSON file handling
from pathlib import Path  # Path operations
```

**Why these?**
- Standard library (no dependencies)
- argparse: Professional CLI interface
- json: Metadata storage
- os/pathlib: File system operations

#### Custom Module Imports:
```python
from src.preprocess import load_image, save_image, calculate_capacity
from src.security import prepare_data_for_embedding, recover_data_from_extraction
from src.dct_engine import embed_message, extract_message
from src.evaluation import evaluate_imperceptibility, print_metrics, apply_attack
```

**Why this structure?**
- Modular design
- Separation of concerns
- Easy to maintain
- Reusable components

---

## Technology Summary Table

| Technology | Files Used In | Purpose | Why This Technology |
|------------|---------------|---------|---------------------|
| **OpenCV** | preprocess.py, dct_engine.py, evaluation.py | Image I/O, DCT operations, color conversion | Industry standard, optimized |
| **NumPy** | All source files | Array operations, math, statistics | Efficient, foundation of scientific Python |
| **Cryptography** | security.py | AES encryption, key derivation | Secure, standard, well-maintained |
| **Reed-Solomon** | security.py | Error correction | Can correct multiple errors, standard |
| **Scikit-image** | evaluation.py | Quality metrics (PSNR, SSIM) | Standard implementations |
| **Hashlib** | dct_engine.py | Password hashing for seed | Standard library, no dependencies |
| **argparse** | main.py | CLI interface | Standard library, professional |
| **json** | main.py | Metadata storage | Standard library, human-readable |

---

## Algorithm Locations

### DCT Algorithm
- **File**: `src/dct_engine.py`
- **Function**: `embed_in_block()`, `extract_from_block()`
- **Library**: OpenCV (`cv2.dct()`, `cv2.idct()`)
- **Lines**: ~131, ~138, ~160

### QIM Algorithm
- **File**: `src/dct_engine.py`
- **Function**: `quantization_index_modulation()`, `extract_bit_from_coefficient()`
- **Library**: NumPy (for rounding, modulo)
- **Lines**: ~75-99, ~101-115

### AES Encryption
- **File**: `src/security.py`
- **Function**: `aes_encrypt()`, `aes_decrypt()`
- **Library**: Cryptography (`Cipher`, `algorithms.AES`, `modes.GCM`)
- **Lines**: ~41-69, ~71-99

### PBKDF2 Key Derivation
- **File**: `src/security.py`
- **Function**: `derive_key()`
- **Library**: Cryptography (`PBKDF2HMAC`)
- **Lines**: ~19-38

### Reed-Solomon ECC
- **File**: `src/security.py`
- **Function**: `apply_reed_solomon()`, `decode_reed_solomon()`
- **Library**: reedsolo (`RSCodec`)
- **Lines**: ~101-118, ~120-144

### Block Selection
- **File**: `src/preprocess.py`
- **Function**: `select_embedding_blocks()`
- **Library**: NumPy (`np.random.RandomState`)
- **Lines**: ~91-141

### Quality Metrics
- **File**: `src/evaluation.py`
- **Function**: `calculate_psnr()`, `calculate_ssim()`
- **Library**: Scikit-image (`psnr`, `ssim`)
- **Lines**: ~13-37

---

## Data Flow Through Technologies

### Embedding Flow:
```
Secret Message (bytes)
    ↓
Cryptography Library (AES-256-GCM encryption)
    ↓
Reed-Solomon Library (Error correction)
    ↓
NumPy (Convert to bits)
    ↓
OpenCV (DCT on blocks)
    ↓
NumPy (QIM algorithm)
    ↓
OpenCV (IDCT on blocks)
    ↓
OpenCV (Save stego image)
```

### Extraction Flow:
```
Stego Image (PNG)
    ↓
OpenCV (Load image, DCT on blocks)
    ↓
NumPy (QIM extraction, bits to bytes)
    ↓
Reed-Solomon Library (Error correction)
    ↓
Cryptography Library (AES-256-GCM decryption)
    ↓
Original Message (bytes)
```

---

## Why Each Technology Choice?

### OpenCV
- **Alternative**: PIL/Pillow
- **Why OpenCV**: Better DCT support, optimized C++ backend, industry standard

### NumPy
- **Alternative**: Python lists
- **Why NumPy**: 100x faster for array operations, memory efficient

### Cryptography Library
- **Alternative**: pycrypto, pynacl
- **Why Cryptography**: Most secure, well-maintained, standard algorithms

### Reed-Solomon (reedsolo)
- **Alternative**: Custom implementation
- **Why reedsolo**: Well-tested, efficient, standard implementation

### Scikit-image
- **Alternative**: Custom PSNR/SSIM implementation
- **Why Scikit-image**: Standard, accurate, well-tested

---

**This mapping shows exactly where each technology is used and why it was chosen.**

