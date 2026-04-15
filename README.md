# 🚧 Automated Road Damage Classification Using Deep Learning

## 📌 Project Overview

This project is a **Deep Learning-based web application** that detects and classifies road damage from images into:

* 🕳️ Pothole
* 🪨 Crack
* ⚙️ Manhole

The system uses **CNN and Transfer Learning models** and is deployed using **Streamlit** for real-time predictions.

---

## 🎯 Objectives

* Automate road damage detection
* Reduce manual inspection effort
* Improve road safety and maintenance planning
* Provide real-time predictions via web app

---

## 🧠 Technologies Used

* **Programming:** Python
* **Deep Learning:** TensorFlow, Keras
* **Models:** CNN, MobileNetV2, ResNet50, EfficientNetB0
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Others:** NumPy, PIL, Scikit-learn

---

## 📂 Project Structure

```
Road-Damage-Classification/
│
├── data/                         # Raw dataset
├── processed_dataset/            # Train/Val/Test split
│
├── model_training.py             # Training script
├── streamlit_app.py              # Web application
│
├── final_model_here.h5           # Trained model
├── README.md                     # Project documentation
```

---

## 📊 Dataset

* Source: RDD2020 / RDD2022 Dataset
* Input: Road images (real-world conditions)
* Classes:

  * Class 0 → Pothole
  * Class 1 → Crack
  * Class 2 → Manhole

### 🔧 Preprocessing

* Image resizing (224×224)
* Normalization (0–1)
* Data augmentation (flip, rotation, zoom)
* Class imbalance handling (class weights)

---

## ⚙️ Model Development

* Built a **baseline CNN model**
* Implemented **transfer learning models**:

  * MobileNetV2 (Best Model ✅)
  * ResNet50
  * EfficientNetB0

### 🔄 Training Features

* Early stopping
* Fine-tuning
* Data augmentation

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🌐 Web Application Features

* 📸 Upload road image
* 🤖 Real-time prediction
* 📊 Confidence score
* 📉 Probability visualization
* 🛠️ Damage-specific recommendations

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Training (Optional)

```bash
python model_training.py
```

### 3️⃣ Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 💡 Example Output

* **Detected Damage:** Pothole
* **Confidence:** 92.45%
* **Recommendation:** Immediate repair required

---

## 📌 Business Use Cases

* 🏙️ Smart City Monitoring
* 🚧 Road Maintenance Planning
* 🚗 Transportation Safety
* 📱 Public Reporting Systems

---

## 🚀 Future Enhancements

* Grad-CAM visualization (Explainable AI)
* Mobile app integration
* Real-time video detection
* Cloud deployment (AWS / Azure / GCP)

---

## ⭐ Conclusion

This project demonstrates how **Deep Learning + Web Deployment** can be used to build a **smart, real-time road damage detection system**, contributing to safer and smarter infrastructure.

