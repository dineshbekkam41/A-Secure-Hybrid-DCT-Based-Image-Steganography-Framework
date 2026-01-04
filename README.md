
# 🔐 A Secure Hybrid DCT-Based Image Steganography Framework

## Using AES-GCM Encryption and Reed–Solomon Error Correction

A robust and secure image steganography framework that integrates **Discrete Cosine Transform (DCT)**, **AES-GCM encryption**, and **Reed–Solomon (RS) error-correction** to achieve **confidentiality, integrity, robustness, and imperceptibility** in digital image communication.

---

## 📌 Project Description

In the digital era, secure transmission of sensitive information over open networks is a major challenge. While cryptography protects data content, it exposes the presence of encrypted information. **Steganography** addresses this limitation by hiding secret data within digital media such as images.

This project proposes a **hybrid image steganography framework** that:

* Encrypts secret data using **AES-GCM** for confidentiality and integrity
* Applies **Reed–Solomon (RS) error-correction** to handle data loss and noise
* Embeds encrypted data into images using **DCT-based frequency domain embedding**
* Ensures robustness against compression, noise, and image processing attacks

---

## 🎯 Objectives

* To securely hide sensitive information inside digital images
* To ensure **data confidentiality and integrity** using AES-GCM
* To improve robustness using **Reed–Solomon error-correction**
* To achieve high imperceptibility using **DCT-based embedding**
* To resist common image processing and transmission attacks

---

## 🧠 System Architecture

The proposed framework consists of the following stages:

1. **Secret Data Encryption**

   * AES-GCM encrypts the secret message
   * Provides confidentiality and authentication

2. **Error-Correction Encoding**

   * Reed–Solomon coding adds redundancy
   * Enables accurate recovery even if data is partially corrupted

3. **DCT-Based Embedding**

   * Cover image is transformed into the frequency domain
   * Encrypted and encoded data is embedded in mid/high-frequency DCT coefficients

4. **Stego Image Generation**

   * Inverse DCT reconstructs the final stego image

5. **Extraction and Recovery**

   * DCT extraction → RS decoding → AES-GCM decryption

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **OpenCV**
* **PyCryptodome (AES-GCM)**
* **Reed–Solomon Error-Correction Library**
* **Jupyter Notebook**

---

## 📊 Evaluation Metrics

The performance of the system is evaluated using:

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**
* **MSE (Mean Squared Error)**
* **RMSE (Root Mean Squared Error)**
* **Bit Error Rate (BER)**

These metrics ensure high image quality and reliable data extraction.

---

## 📁 Project Structure

```
├── data/
│   ├── cover_images/
│   ├── secret_data/
│
├── encryption/
│   ├── aes_gcm.py
│
├── error_correction/
│   ├── reed_solomon.py
│
├── steganography/
│   ├── dct_embed.py
│   ├── dct_extract.py
│
├── evaluation/
│   ├── metrics.py
│
├── notebooks/
│   ├── demo.ipynb
│
├── results/
│
├── README.md
└── requirements.txt
```

---



## 🔐 Security Highlights

* **AES-GCM** ensures:

  * Strong encryption
  * Message authentication
  * Protection against tampering

* **Reed–Solomon coding**:

  * Corrects transmission and compression errors
  * Improves robustness

* **DCT embedding**:

  * Resistant to JPEG compression
  * Less vulnerable to statistical attacks

---

## 📈 Results Summary

* High visual quality of stego images
* Successful recovery of secret data even after noise and compression
* Improved robustness compared to basic LSB-based methods

---

## ⚠️ Limitations

* Computational overhead due to encryption and error correction
* Payload capacity is limited compared to pure spatial-domain methods

---

## 🔮 Future Enhancements

* Extension to **video steganography**
* Integration with **deep learning–based steganalysis detection**
* Adaptive embedding based on image texture
* Optimization for real-time applications

---

## 👨‍🎓 Academic Use

This project is suitable for:

* Final-year / capstone projects
* Research extensions
* Secure communication demonstrations

---

## 📜 License

This project is intended for **academic and educational use only**.

