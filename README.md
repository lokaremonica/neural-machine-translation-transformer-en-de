# Neural Machine Translation (English ↔ German) with Transformers

## 📖 Overview

**Neural Machine Translation (NMT)** is a deep learning approach for automatically translating text from one language to another. Unlike traditional phrase-based or rule-based translation systems, NMT uses neural networks to learn the meaning and structure of entire sentences, enabling more fluent and contextually accurate translations.

This project implements an **English ↔ German translation system** using a **Transformer** model in PyTorch. It covers data preprocessing, model training, evaluation, and inference with greedy decoding. The dataset used is **Multi30k**, a standard benchmark for translation tasks.

---

## ✨ Features

* End-to-end NMT system for **EN→DE** and **DE→EN** translation.
* Tokenization with **spaCy** and vocabulary creation with special tokens (`<bos>`, `<eos>`, `<pad>`).
* Implementation of **Transformer encoder–decoder** with positional encoding.
* **Greedy decoding** for inference and **back-translation checks** for quality assessment.
* Modular code with separate scripts for data processing, model architecture, training, and inference.
* Results tracking with train/validation loss logs and sample translations.

---

## 🛠 Tech Stack

* **Python** 3.10+
* **PyTorch** (Transformer API)
* **TorchText** for dataset handling
* **spaCy** for tokenization
* **TQDM** for training progress

---

## 🚀 Training

Example: Train English → German

```bash
python -m src.train \
  --src_lang en --tgt_lang de \
  --epochs 18 --batch_size 128 \
  --emb 512 --nhead 8 --ffn 512 \
  --enc_layers 3 --dec_layers 3
```

**Sample results:**

* **EN→DE**: Train loss ≈ **1.07**, Val loss ≈ **2.08**
* **DE→EN**: Train loss ≈ **0.97**, Val loss ≈ **1.93**

---

## 💡 Inference

Example translation:

```python
from src.infer import translate
print(translate(model, "The dog is running in the park."))
# Output: "Der Hund läuft im Park."
```

---

## 🔍 Back-Translation Check

Translate EN→DE, then back DE→EN to check meaning preservation.
Example:

* EN: "The dog is running in the park."
* DE: "Der Hund läuft im Park."
* Back to EN: "The dog is running in the park." 
