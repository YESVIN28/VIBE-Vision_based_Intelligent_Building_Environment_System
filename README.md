# VIBE (Vision based Intelligent Building Environment System ) 
<h1 align="center">VIBE</h1>  
<br />
<p align="center">
  <strong>Vision-based Intelligent Building Environment</strong><br>
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
  <img src="https://i.imgur.com/HowF6aM.png" width="450">
</p>

---

## 🌐 Introduction

**VIBE** is a real-time intelligent environmental automation system designed for smart buildings. It uses crowd detection and density estimation through **YOLOv8**, **DeepSORT**, and a novel **Multi-Scale DensityResNet**. These insights control lighting, HVAC, and display systems via IoT integration on a Raspberry Pi.

It reduces energy consumption by up to **28.5%**, maintains thermal comfort within **±0.5°C**, and runs efficiently on edge platforms.

---

## ⚙️ Features

- 🧍 Person Detection using YOLOv8
- 👥 Crowd Density Estimation with ResNet + CBAM + ASPP
- 🔁 DeepSORT Multi-object Tracking with Appearance Re-ID
- 🌡️ IoT-based Environmental Monitoring (DHT11)
- 🧠 Predictive Decision Engine for Environmental Control
- 🌐 Real-Time Web Dashboard (Flask + JS)
- ⚡ 45 FPS Real-time Processing on Raspberry Pi 4
- 🔒 Privacy-by-Design Architecture (Edge Processing)

---

## 🧰 Tech Stack

| Category               | Technologies |
|------------------------|--------------|
| Programming Language   | Python 3, HTML/CSS, JavaScript |
| Frameworks             | Flask, PyTorch, OpenCV |
| ML Models              | YOLOv8, ResNet, DeepSORT, EfficientNet |
| Edge Devices           | Raspberry Pi 4 |
| Sensors                | DHT11, GPIO |
| Deployment             | Local server / Raspberry Pi |
| Visualization          | WebRTC, Chart.js |
| Training Dataset       | ShanghaiTech, JHU-CROWD++, UCF-QNRF |

---

## 📊 System Architecture

<p align="center">
  <img src="https://i.imgur.com/IkSnFRL.png" width="750">
</p>

---

## 🚀 Installation

### 🔧 Prerequisites
- Python 3.8+
- Raspberry Pi 4 (8GB RAM recommended)
- DHT11 Sensor connected via GPIO
- A working webcam

### 📦 Install Dependencies

```bash
git clone https://github.com/yourusername/VIBE.git
cd VIBE
pip install -r requirements.txt
