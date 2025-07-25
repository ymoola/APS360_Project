# 🦷 Oral Disease Classification App

A computer vision web app built with **ViT (Vision Transformer)** and an **ANN classifier** to detect oral diseases from mouth images.

Developed for the APS360 course project at the University of Toronto.

---

## 🔍 Overview

This Streamlit-powered app allows users to upload images of mouths or teeth and receive predictions for the following oral diseases:

- Calculus  
- Caries  
- Gingivitis  
- Hypodontia  
- Ulcer  
- Tooth Discoloration  

The pipeline uses:

- 🔎 **ViT** (pretrained on ImageNet) to extract deep features  
- 🧠 **ANN classifier** trained on extracted ViT embeddings  
- 🌐 Deployed using **Streamlit Cloud**

---

## 🚀 Run the App

### 🧑‍💻 Locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/ymoola/APS360_Project.git
   cd APS360_Project/oral_disease_app
   
2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   
3. **Run App**

   ```bash
   streamlit run app.py

  💡 Make sure the model file model_VIT_bs32_lr0.0001_epoch24 is present in the root of the repo or update the path in predict.py.

---

## ☁️ Online (Streamlit Cloud)
App is deployed here:
https://oral-disease.streamlit.app

---

## 🧠 Model Architecture

ViT Base (Patch16, 224) from timm is used for feature extraction.
A lightweight fully connected layer (ANN) classifies extracted embeddings into 6 oral disease categories.

---

## 📂 Project Structure

```bash
APS360_Project/
├── oral_disease_app/
│   ├── app.py
│   ├── model/
│   │   ├── vit_feature_extractor.py
│   │   ├── ann_classifier.py
│   │   └── model_VIT_bs32_lr0.0001_epoch24
│   ├── utils/
│   │   └── predict.py
│   ├── requirements.txt
│   └── README.md



