# 🤖 Dynamic Hand Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-18.0+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)

> **Real-time gesture recognition system combining static and dynamic hand gestures with audio-visual learning features for enhanced sign language communication.**

> **🔴 Live Demo (Azure Kubernetes Service):** [http://48.194.106.0](http://48.194.106.0) *(Please allow a few seconds for the pods to wake up on first load)*

## 🎯 Overview

This comprehensive gesture recognition system leverages deep learning to detect and classify both static hand gestures (numbers, alphabets) and dynamic gestures (words, commands) in real-time. Built with a modern web interface, the system includes innovative features like speech-to-GIF mapping and visual learning capabilities, making it an effective tool for sign language education and accessibility.

## ✨ Key Features

### 🖐️ **Dual Gesture Recognition**
- **Static Gestures**: Numbers (1-9) & Alphabets (A-Z) using ANN with 87 landmark features
- **Dynamic Gestures**: Words and commands using LSTM with face+hand+body landmarks (126+ features)

### 🌐 **Full-Stack Web Application**
- **Frontend**: React.js with real-time webcam integration
- **Backend**: Flask API for model inference and data processing
- **Real-time Predictions**: Live confidence scores and gesture classification

### 🎵 **Audio-Visual Learning**
- **Speech Recognition**: Voice-to-text processing
- **GIF Database**: Mapped visual responses for enhanced learning
- **Multimodal Interface**: Gesture + Audio input for comprehensive communication

### 🔥 **Advanced ML Pipeline**
- **Computer Vision**: OpenCV + MediaPipe for robust landmark extraction
- **Deep Learning**: Custom ANN and LSTM architectures
- **Real-time Processing**: Optimized for low-latency predictions

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │   Flask Backend │    │  ML Models      │
│                 │    │                 │    │                 │
│ • Webcam Stream │◄──►│ • API Endpoints │◄──►│ • Static ANN    │
│ • Real-time UI  │    │ • Model Loading │    │ • Dynamic LSTM  │
│ • Audio Input   │    │ • GIF Mapping   │    │ • MediaPipe     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Interface│    │   Data Pipeline │    │   CV Processing │
│                 │    │                 │    │                 │
│ • Predictions   │    │ • Preprocessing │    │ • Landmark Ext. │
│ • Confidence    │    │ • Augmentation  │    │ • Normalization │
│ • GIF Display   │    │ • Validation    │    │ • Feature Eng.  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

### **Machine Learning & AI**
| Technology | Purpose | Version |
|------------|---------|---------|
| TensorFlow/Keras | Deep Learning Framework | 2.8+ |
| MediaPipe | Hand/Face/Body Landmark Detection | 0.8+ |
| OpenCV | Computer Vision Processing | 4.5+ |
| NumPy | Numerical Computing | 1.21+ |
| Scikit-learn | ML Utilities | 1.0+ |

### **Web Development**
| Technology | Purpose | Version |
|------------|---------|---------|
| React.js | Frontend Framework | 18.0+ |
| Flask | Backend API | 2.0+ |
| JavaScript/ES6 | Frontend Logic | - |
| HTML5/CSS3 | UI Structure & Styling | - |
| Axios | HTTP Client | 0.27+ |

### **Cloud & DevOps (New!)**
| Technology | Purpose | 
|------------|---------|
| Azure Kubernetes Service (AKS) | Scalable Cloud Orchestration | 
| Docker & Docker Compose | Containerization | 
| Azure Container Registry (ACR) | Cloud Image Storage | 
| Weights & Biases (W&B) | Live Training Metrics & Logging | 
| MLflow | Model Registry & Versioning | 
| GitHub Actions | CI/CD Pipelines | 

### **Data & Storage**
| Technology | Purpose | Version |
|------------|---------|---------|
| JSON | Data Serialization | - |
| CSV | Dataset Storage | - |
| Local Storage | Model & GIF Storage | - |

## 📊 Model Performance

### **Static Gesture Recognition (ANN)**
- **Architecture**: Multi-layer Neural Network
- **Input Features**: 87 hand landmarks (MediaPipe)
- **Classes**: 35 (Numbers 1-9, Alphabets A-Z)
- **Accuracy**: ~92% on test set
- **Inference Time**: <10ms per prediction

### **Dynamic Gesture Recognition (LSTM)**
- **Architecture**: Bidirectional LSTM + Dense layers
- **Input Features**: 126+ (face+hand+body landmarks)
- **Sequence Length**: 20-30 frames
- **Classes**: 15+ dynamic gestures/words
- **Accuracy**: ~88% on validation set
- **Inference Time**: <50ms per sequence

## 🚀 Quick Start

### **Prerequisites**
```bash
# Python 3.8+
python --version

# Node.js 16+
node --version
npm --version
```

### **Installation**

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/gesture-recognition-system.git
cd gesture-recognition-system
```

2. **Backend Setup**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

3. **Frontend Setup**
```bash
cd frontend
npm install
npm start
```

4. **Access the Application**
```
Frontend: http://localhost:3000
Backend API: http://localhost:5000
```

### **Cloud Deployment (AKS)**
This project is fully configured for cloud deployment on Azure Kubernetes Service.
```bash
# Push images to Azure Container Registry
docker push <acr-name>.azurecr.io/backend:latest
docker push <acr-name>.azurecr.io/frontend:latest

# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
```

## 📂 Project Structure

```
gesture-recognition-system/
├── 📁 frontend/                 # React.js web application
│   ├── public/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── services/            # API services
│   │   └── styles/              # CSS styling
│   └── package.json
│
├── 📁 backend/                  # Flask API server
│   ├── app.py                   # Main Flask application
│   ├── models/                  # Trained ML models
│   │   ├── static_model.h5      # ANN for static gestures
│   │   └── dynamic_model.h5     # LSTM for dynamic gestures
│   ├── utils/                   # Utility functions
│   │   ├── preprocessing.py     # Data preprocessing
│   │   ├── landmark_extraction.py
│   │   └── audio_processing.py
│   ├── gifs_database/           # GIF files and mappings
│   └── requirements.txt
│
├── 📁 data/                     # Dataset and training data
│   ├── static/                  # Static gesture datasets
│   ├── dynamic/                 # Dynamic gesture datasets
│   └── processed/               # Preprocessed data
│
├── 📁 notebooks/                # Jupyter notebooks
│   ├── static_training.ipynb    # Static model training
│   ├── dynamic_training.ipynb   # Dynamic model training
│   └── data_analysis.ipynb      # Data exploration
│
├── 📁 scripts/                  # Utility scripts
│   ├── data_collection.py       # Data collection tools
│   ├── model_training.py        # Training pipelines
│   └── evaluation.py            # Model evaluation
│
└── 📄 README.md                 # Project documentation
```

## 🎮 Usage Guide

### **1. Real-time Gesture Recognition**
1. Click "Start Prediction" on the web interface
2. Allow camera permissions
3. Show static gestures (numbers/letters) or dynamic gestures (words)
4. View real-time predictions with confidence scores

### **2. Audio-to-GIF Learning**
1. Click the microphone button
2. Speak a word clearly
3. Watch as the system displays the corresponding GIF
4. Use for visual learning and sign language practice

### **3. Data Collection Mode**
1. Run the data collection script
2. Follow on-screen instructions for recording gestures
3. Ensure proper lighting and hand visibility
4. Collect diverse samples for robust training

## 🧠 Model Details

### **Static Gesture Model (ANN)**
```python
# Architecture Overview
Input Layer (87 features) 
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (35 classes, Softmax)
```

### **Dynamic Gesture Model (LSTM)**
```python
# Architecture Overview
Input Layer (sequence_length, 126+ features)
    ↓
LSTM Layer (128 units, return_sequences=True)
    ↓
LSTM Layer (64 units)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (num_classes, Softmax)
```

## 🔬 Training Process

### **Data Collection**
- **Static**: 50+ samples per class (35 classes)
- **Dynamic**: 30+ sequences per gesture (15+ gestures)
- **Augmentation**: Rotation, scaling, noise injection
- **Validation**: 80/20 train-test split

### **Training Configuration**
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Regularization**: Dropout, L2 regularization

## 🌟 Innovations

### **1. Multi-Modal Landmark Detection**
- Traditional approach: Hand landmarks only
- **Our Innovation**: Face + Hand + Body landmarks for better reference points
- **Result**: Improved accuracy for dynamic gestures

### **2. Audio-Visual Integration**
- **Speech-to-GIF Mapping**: Voice commands trigger visual responses
- **Educational Focus**: Enhanced learning for sign language students
- **Accessibility**: Multi-sensory communication support

### **3. Real-time Web Interface**
- **Seamless Integration**: Direct webcam access in browser
- **Live Feedback**: Real-time confidence scores and predictions
- **User-Friendly**: One-click prediction activation

## 🎯 Applications

- **🏫 Education**: Sign language learning and practice
- **♿ Accessibility**: Communication aid for hearing-impaired individuals
- **🎮 Gaming**: Gesture-based game controls
- **🏠 Smart Home**: Hands-free device control
- **👨‍💻 Development**: Computer vision research and prototyping

<div align="center">

### 🌟 If you found this project helpful, please give it a star! ⭐

[![GitHub stars](https://img.shields.io/github/stars/yourusername/gesture-recognition-system.svg?style=social&label=Star)](https://github.com/yourusername/gesture-recognition-system)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/gesture-recognition-system.svg?style=social&label=Fork)](https://github.com/yourusername/gesture-recognition-system/fork)

**Made with ❤️ for accessibility and innovation in human-computer interaction**

</div>
