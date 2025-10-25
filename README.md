# 🛡️ Network Intrusion Detection System (NIDS)

## 📌 What is NIDS?  
A **Network Intrusion Detection System (NIDS)** is a security solution that monitors network traffic in real-time to detect **malicious activities, attacks, or policy violations**.  
It analyzes traffic patterns and flags anomalies or known attack signatures.  

### Common Attack Categories:
- **DoS (Denial of Service):** Flooding a system to make it unavailable (e.g., `neptune`, `smurf`).  
- **Probe:** Scanning/surveillance to gather info before an attack (e.g., `nmap`, `portsweep`).  
- **R2L (Remote to Local):** Remote attacker gaining unauthorized local access (e.g., `guess_passwd`).  
- **U2R (User to Root):** Local user escalating privileges to root (e.g., `buffer_overflow`).  
- **Normal:** Legitimate, benign traffic.  

---

## 📌 About This Project  
This project implements a **deep learning–based NIDS** using the **NSL-KDD dataset**.  
- Hybrid **CNN–BiLSTM model with attention mechanism** for sequential feature learning.  
- Handles **class imbalance** using weighted training.  
- Preprocessing pipeline includes **feature scaling, one-hot encoding, and label encoding**.  
- Supports **training, testing, and inference** with reusable modules.  

---

## 🚀 Features  
- End-to-end **training pipeline** (`train_pipline.py`)  
- Modular code for **data processing** (`data_processor.py`) and **model architecture** (`model.py`)  
- **Custom attention layer** for better feature extraction  
- Saves **preprocessing artifacts** (`preprocessor.joblib`, `label_encoder.joblib`) for consistent predictions  
- Supports **batch testing** on `KDDTest+` dataset  
- Prototype **frontend (Streamlit/Flask)** for real-time intrusion detection  

---


