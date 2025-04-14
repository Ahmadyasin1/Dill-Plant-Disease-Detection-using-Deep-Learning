# 🌿 Dill Plant Disease Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

A deep learning-powered system to detect **diseases in dill plants** from leaf images. This project uses state-of-the-art convolutional neural networks (CNNs) to classify the health condition of dill leaves — helping farmers and agricultural experts make faster, data-driven decisions.

---

## 🌱 Project Overview

This project tackles **plant disease diagnosis** using **image classification**. By training multiple deep learning models on a custom dataset of dill leaf images, the system can automatically detect and classify various diseases based on visual patterns.

The trained models can:
- 🧠 Learn leaf textures and symptoms via CNNs
- 📸 Predict the disease category from input images
- 📊 Compare performance across multiple architectures
- 🖥️ Be integrated into real-time mobile/web solutions

---

## 📂 Dataset

- **Type**: Image dataset of dill plant leaves
- **Categories**: Healthy, Powdery Mildew, Leaf Blight, Rust, etc.
- **Format**: JPG/PNG images organized in class-based folders
- **Preprocessing**: Image resizing, normalization, augmentation (rotation, flip, zoom, etc.)

---

## 🧠 Models Trained

- ✅ Custom CNN (from scratch)
- ✅ MobileNetV2 (transfer learning)
- ✅ ResNet50
- ✅ EfficientNetB0
- ✅ VGG16

All models were trained, validated, and tested on the same image dataset for performance comparison.

---

## 🧪 Evaluation Metrics

- Accuracy
- Precision & Recall
- F1 Score
- Confusion Matrix
- Training & Validation Loss
- ROC-AUC (if applicable)

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib
- **Tools**: Jupyter Notebook, Google Colab / VS Code
- **Model Deployment**: [optional: Flask, Streamlit, TensorFlow Lite]

---

## 🔮 Future Enhancements
📲 Convert model for mobile use (TensorFlow Lite)

☁️ Deploy as an online detection web app (Streamlit / Flask)

🧠 Integrate Grad-CAM for model explainability

🌐 Build multi-language farmer-friendly app

📷 Add real-time camera integration

---

## 🤝 Contributing
We welcome contributions from the community! Feel free to:

Fork the repo

Create a feature branch

Open a pull request

Or suggest improvements via Issues

---

## 📬 Contact
Ahmad Yasin
📧 Email: AhmadYasin.info@gmail.com
🔗 LinkedIn: www.linkedin.com/in/mian-ahmad-yasin
🌐 Portfolio: https://ahmadyasin.vercel.app/

---

## ⭐ Support
If this project helped you or inspired you, please consider giving it a ⭐ star on GitHub. It means a lot!

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/dill-plant-disease-detection.git
cd dill-plant-disease-detection

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook or Python script
jupyter notebook Dill_Plant_Disease_Detection.ipynb
