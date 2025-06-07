# 🧠 Brain Tumor Detector Web App

This is a deep learning-based web application for detecting brain tumors using MRI images. The model is trained on MRI scan datasets, converted to TensorFlow Lite (`.tflite`) format for lightweight inference, and deployed using a Flask web server with a simple HTML/CSS/JS frontend.

## 🚀 Demo
📌 [Link to live demo (optional if hosted)](https://yourappurl.com)  
📸 Upload an MRI scan to detect if a brain tumor is present.

---

## 🗂️ Project Structure

brain-tumor-detector-web/
│── app.py
├── static/ # CSS and JS files
├── templates/ # HTML files
├── Brain_tumor_model.tflite
├── colab-notebooks/ # Training notebooks (.ipynb)
├── requirements.txt # Python dependencies
├── README.md
└── LICENSE

## ⚙️ How to Run Locally

### 🔹 1. Clone the repository


git clone https://github.com/anubhav9369/brain-tumor-detector-web.git
cd brain-tumor-detector-web


### 🔹 2. Create a virtual environment and install dependencies
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt

### 🔹 3. Run the app
bash
Copy
Edit
cd app
python app.py

