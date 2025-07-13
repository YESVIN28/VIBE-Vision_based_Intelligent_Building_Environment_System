<h1 align="center">VIBE</h1>  
<br />
<p align="center">
  <strong>Vision-based Intelligent Building Environment System</strong><br>
  Real-time Crowd-Aware Smart Environmental Automation
</p>

<p align="center">
  <img src="https://img.shields.io/badge/IoT-Enabled-green.svg" />
  <img src="https://img.shields.io/badge/Computer%20Vision-YOLOv8-blue.svg" />
  <img src="https://img.shields.io/badge/Machine%20Learning-ResNet-orange.svg" />
  <img src="https://img.shields.io/badge/Edge%20Computing-RPi%204-critical.svg" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" />
</p>

---

<p align="center">
  <img src="https://github.com/YESVIN28/VIBE-Vision_based_Intelligent_Building_Environment_System/blob/main/generated-svg-image.svg?raw=true" width="500" />
</p>

---

## 🌐 Introduction

**VIBE** is a real-time intelligent environmental automation system designed for smart buildings. It uses **YOLOv8**, **DeepSORT**, and a novel **Multi-Scale DensityResNet** to detect crowd density, control lighting, HVAC, and display systems through IoT devices.

🔋 Saves up to **28.5% energy**, operates at **45 FPS**, and maintains comfort within **±0.5°C**.

---

## ⚙️ Key Features

- 🧍 YOLOv8 Person Detection  
- 👥 Crowd Density Estimation (ResNet + CBAM + ASPP)  
- 🔁 DeepSORT Multi-Object Tracking  
- 🌡️ DHT11-based IoT Environmental Monitoring  
- 🧠 ML-Powered Adaptive Decision Engine  
- 🌐 Real-Time Web Dashboard  
- ⚡ 45 FPS on Raspberry Pi 4  
- 🔒 Edge-Processed, Privacy-Centric Design  

---

## 🧰 Tech Stack

| Category             | Technologies |
|----------------------|--------------|
| Languages            | Python 3, HTML/CSS, JavaScript |
| Frameworks           | Flask, PyTorch, OpenCV |
| Models Used          | YOLOv8, ResNet, DeepSORT |
| Edge Devices         | Raspberry Pi 4 |
| Sensors              | DHT11 (Temperature, Humidity) |
| Visualization        | WebRTC, Chart.js |
| Datasets             | ShanghaiTech, JHU-CROWD++, UCF-QNRF |

---

## 📊 System Workflow

<p align="center">
  <img src="https://github.com/YESVIN28/VIBE-Vision_based_Intelligent_Building_Environment_System/blob/main/IMAGE_FLOW.PNG?raw=true" width="1000" />
</p>

---

## 📱 Dashboard Overview

- 👁️ Live Feed with bounding boxes and density overlays  
- 📈 Sensor Graphs for real-time temperature and humidity  
- ⚙️ Control Panel to toggle fans, lights, and display  
- 🔔 Smart Alerts for crowd density and environmental thresholds  

---

## 📈 Results & Performance

| Metric                     | Value        |
|----------------------------|--------------|
| Crowd Estimation Accuracy  | 91.2%        |
| YOLOv8 mAP@0.5             | 94.2%        |
| Identity Preservation      | 91.8%        |
| Mean Absolute Error (MAE)  | 6.9 (Part A) |
| Real-Time FPS              | 45           |
| Energy Saved               | 28.5%        |

---

## 🧪 Evaluation Metrics

- ✅ Mean Average Precision (**mAP**)  
- ✅ Mean Absolute Error (**MAE**)  
- ✅ **F1 Score**  
- ✅ Identity Re-identification Accuracy  

---

## 💡 Use Cases

- 🏢 Smart Office Automation  
- 🛍️ Malls & Shopping Centers  
- 🚉 Public Transit Stations  
- 🏥 Hospital Monitoring  
- 🏫 School and Lecture Hall Management  

---

## 🔑 Model Weights

- Download YOLOv8 weights from [Ultralytics](https://github.com/ultralytics/yolov8)
- Get ResNet-based density estimation model (custom-trained)
- Use DeepSORT checkpoint with EfficientNet-B2

📁 Place all model weights in the `/models/` directory.

---

## 🔧 Installation Guide

### **Prerequisites**
- Python 3.8+
- Raspberry Pi 4 (8GB RAM recommended)
- DHT11 Sensor (connected via GPIO)
- USB Webcam

--- 

## 🛠️ Setup
```bash```
### Clone the repository
git clone https://github.com/YESVIN28/VIBE-Vision_based_Intelligent_Building_Environment_System.git
cd VIBE-Vision_based_Intelligent_Building_Environment_System

### Install dependencies
pip install -r requirements.txt

---

## 🖥️ Run the Application
`bash`

### Start the Flask server
python app.py

### 🔗 Then open your browser and visit:
http://localhost:5000

---

## 📬 Feedback & Contributions

- 💬 Found a bug or feature idea? → [Open an issue](https://github.com/YESVIN28/VIBE-Vision_based_Intelligent_Building_Environment_System/issues)  
- 🤝 Want to contribute? → Fork the repo or create a pull request  
- 📧 Email: **yesvinveluchamy@gmail.com**

---

## 👏 Acknowledgements

Special thanks to:

- 🎓 **Dr. M. Sridevi**, Project Mentor – National Institute of Technology, Tiruchirappalli  
- [Ultralytics](https://github.com/ultralytics) for YOLOv8  
- [JetBrains](https://www.jetbrains.com/) for open source support  
- OpenCV and PyTorch community  

---

## 📄 License

This project is licensed under the **MIT License**.  
Refer to the [LICENSE](LICENSE) file for more information.

---

## 📎 Appendix

- 📁 Datasets: **ShanghaiTech**, **UCF-QNRF**, **JHU-CROWD++**  
- 🧠 Training setup and configs included  
- 📊 Performance benchmarks and ablation studies  
- 🔗 Related work: [VIBE Repository](https://github.com/YESVIN28/VIBE)
