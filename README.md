# ❤️ Heart Disease Prediction Web App

A Machine Learning powered web application that predicts the risk of heart disease based on medical parameters.

Built using **Python, Flask, Scikit-learn, and Bootstrap (Glassmorphism UI).**

---

## 🚀 Project Overview

This application allows users to input 13 medical parameters and predicts whether a person is at **Low Risk** or **High Risk** of heart disease using a trained **Random Forest Classifier** model.

The project demonstrates:

- Machine Learning model training
- Model serialization using Pickle
- Flask backend integration
- Responsive frontend using Bootstrap
- Modern transparent UI design

---

## 🧠 Machine Learning Model

- Algorithm: **Random Forest Classifier**
- Dataset: Heart Disease dataset
- Features: 13 medical attributes
- Output: Binary Classification (0 = Low Risk, 1 = High Risk)

---

## 📋 Input Parameters

| Feature | Description |
|----------|------------|
| age | Age of patient |
| sex | 0 = Female, 1 = Male |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar (>120 mg/dl) |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |

---

## 🖥️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML5
- CSS3
- Bootstrap 5
- Git & GitHub

---

## 📦 Project Structure

```
Heart-Disease-Prediction-ML/
│
├── app.py
├── model.pkl
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── images/
        └── project.JPG
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/1shubhamSangale1997/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
python app.py
```

### 4️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 🎯 Features

✔ Predict heart disease risk  
✔ Modern transparent glass UI  
✔ Responsive design  
✔ ML model integration with Flask  
✔ Clean project structure  

---

## 🔮 Future Improvements

- Deploy on Render / Railway
- Add prediction probability percentage
- Store prediction history in database
- Add login authentication
- Add API endpoint version

---

## 👨‍💻 Author

**Shubham Sangale**

- Machine Learning Enthusiast  
- QA Automation Engineer  
- Python & Selenium Developer  

GitHub: https://github.com/1shubhamSangale1997  

---

C:\Users\Shubham\AppData\Local\Programs\Python\Python312\python.exe C:/shubham/heart-detector/Heart_disease_detector/app.py
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 109-174-024

## 📌 Disclaimer

This application is for educational and demonstration purposes only.  
It is not intended for real medical diagnosis.
