# 🍫 Roasted Cocoa Classification using TinyML and ESP32

Artificial Intelligence project focused on roasted cocoa classification using Machine Learning, Deep Learning, TensorFlow Lite, TinyML, and ESP32 simulation.

The project was developed in Google Colab for training and Visual Studio Code + PlatformIO + Wokwi for embedded deployment simulation.

---

# 📌 Project Objective

Develop an AI-based system capable of classifying:

- Dry Cocoa
- Burnt Cocoa

using gas sensor data and deploying the trained model on an ESP32 embedded environment.

---

# 🚀 Technologies Used

## Artificial Intelligence & Data Science
- Python
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow
- Keras
- TensorFlow Lite

## Embedded Systems
- ESP32
- TensorFlow Lite Micro
- TinyML
- Arduino Framework

## Development Tools
- Google Colab
- Visual Studio Code
- PlatformIO
- Wokwi Simulator

---

# 📂 Dataset Description

The dataset contains sensor measurements collected from cocoa samples using MQ gas sensors.

## Classes
| Target | Description |
|---|---|
| 0 | Dry Cocoa |
| 1 | Burnt Cocoa |

---

# 🧹 Data Preprocessing

The preprocessing stage included:

- Dataset consolidation
- Null value validation
- Duplicate data analysis
- Duplicate removal
- Feature normalization
- Train/Test split

## Important Finding

More than **7000 duplicated rows** were detected in the dataset.

Duplicated data can produce:
- Overfitting
- Statistical bias
- Artificially high accuracy

To improve model reliability:

```python
df_final = df_final.drop_duplicates()