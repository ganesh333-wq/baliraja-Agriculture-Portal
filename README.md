
# 🌾 Crop Disease and Fertilizer Recommendation System 🌱

## 🔍 Overview

A smart agriculture platform built using **Python Flask**, integrating machine learning and deep learning to detect crop diseases, recommend suitable fertilizers, and provide actionable insights for farmers and agriculture professionals 🧑‍🌾🌿.

***

## ✨ Features

- 🦠 **Crop Disease Detection:** Uses a deep learning model to identify plant diseases from images and provides tailored prevention/cure advice 🖼️🌾.
- 💧 **Fertilizer Recommendation:** Suggests optimal fertilizers based on soil nutrient levels and crop requirements using rule-based logic 🌱🧪.
- 🖥️ **User-Friendly Web Interface:** Responsive design with Bootstrap; easy navigation for submitting crop images and soil details 🖱️.
- 🔧 **Integrated Data Analytics:** This project has built-in tools that check and analyze the collected data carefully. It uses Python programs to read error logs and fix any problems, making sure the data is accurate and trustworthy 🐍📊.
- 💻 **Multi-Environment Support:** Runs on Windows and Ubuntu environments out-of-the-box 💻🖥️.

***

## 📁 Project Structure

| File/Folder         | Purpose                                                       |
|---------------------|--------------------------------------------------------------|
| `model.py`          | Deep learning model for disease detection (ResNet9)           |
| `disease.py`        | Disease information & advice dictionary                       |
| `fertilizer.py`     | Fertilizer advice dictionary (soil nutrients)                 |
| `config.py`         | Configuration for Flask app and environment variables         |
| `requirements.txt`  | Python package dependencies                                   |
| `Runtime.txt`       | Python version info                                           |
| `bootstrap.css`     | Main UI styles (Bootstrap framework)                         |
| `style.css`, `font-awesome.min.css` | Additional styles/icons                     |

***

## 🛠️ Installation

1. **Clone the Repository**  
   ```
   git clone https://github.com/ganesh333-wq/baliraja-Agriculture-Portal.git

   cd baliraja-Agriculture-Portal
   ```

2. **Install Requirements**  
   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**  
   ```
   python app.py
   ```
   - Access the web interface at `localhost:5000` 🌐

***

## 🚀 Usage

- Upload crop images to detect diseases and get cure suggestions 📸🤒.
- Enter soil nutrient data to receive fertilizer recommendations 🌱🧴.
- Analyze and review results with detailed advice for each crop type 📋👍.

***

## 💻 Technology Stack

- **Backend:** Python, Flask, PyTorch, pandas, scikit-learn, numpy 🐍🤖
- **Frontend:** HTML, CSS, Bootstrap, Font Awesome 🎨🖥️
- **Deployment:** Compatible with Windows 🪟🐧

***

## 🙏 Acknowledgments

- Inspired by real-world agricultural needs 🌍🌾.
- Utilizes open-source datasets, research articles, and farmer feedback 📚👩‍🌾.
- Project structure and README style adapted from top data science and ML web app repositories 🧑‍💻📖.

***

