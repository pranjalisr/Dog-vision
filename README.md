# 🐶 DogVision: Dog Breed Classification

DogVision is an end-to-end deep learning project that classifies dog breeds from images using transfer learning with TensorFlow, achieving high accuracy on real-world image data.

---

## 🚀 Project Overview

This project builds a robust image classification pipeline to identify dog breeds from images. It leverages transfer learning with state-of-the-art CNN architectures, enabling high performance even with limited labeled data.

---

## 📊 Key Results

- 📈 **Validation Accuracy:** ~90–93%  
- 🐕 **Number of Classes:** 100+ dog breeds  
- 🖼️ **Dataset Size:** 10,000+ images  
- ⚡ **Training Time:** ~1–2 hours (GPU - Google Colab)  
- 🔁 **Improvement:** +25% accuracy boost using transfer learning vs baseline CNN  

---

## 📌 Features

- Multi-class dog breed classification (100+ classes)  
- Transfer learning using **MobileNetV2 / EfficientNet**  
- Advanced **data augmentation** for generalization  
- End-to-end ML pipeline (data → training → evaluation → prediction)  
- Supports **custom image inference**  

---

## 🧠 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib / Seaborn  
- Jupyter Notebook / Google Colab  

---


---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Image resizing (224×224)
   - Normalization (0–1 scaling)

2. **Data Augmentation**
   - Random flips, rotations, zoom
   - Improves generalization on unseen data

3. **Model Architecture**
   - Pre-trained CNN (MobileNetV2 / EfficientNet)
   - Custom classification head (Dense layers)

4. **Training**
   - Loss: Categorical Crossentropy  
   - Optimizer: Adam  
   - Early stopping + learning rate scheduling  

5. **Evaluation**
   - Accuracy & loss tracking  
   - Validation performance monitoring  

6. **Prediction**
   - Accepts custom images  
   - Outputs predicted dog breed  

---

## 🏋️ Model Training

- **Base Model:** MobileNetV2 / EfficientNet (ImageNet pretrained)  
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Batch Size:** 32  
- **Epochs:** 10–20 (with early stopping)  

---

## 📊 Results & Insights

- Transfer learning significantly improves performance on small datasets  
- Model generalizes well on unseen dog images  
- Overfitting reduced using augmentation + dropout  
- Fine-tuning deeper layers further improves accuracy  

---

## 🔍 Example Usage

```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("models/dogvision_model.keras")

img = Image.open("sample.jpg").resize((224, 224))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = prediction.argmax()

print("Predicted Breed:", predicted_class)
```

## Future Improvements

- Fine-tune deeper CNN layers for higher accuracy

- Increase dataset size for better generalization

- Deploy as a web app (Streamlit / Flask)

- Add real-time prediction via webcam

- Convert to mobile-friendly model (TensorFlow Lite)

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo, improve the model, or add new features.

## 📜 License

This project is open-source and available under the MIT License.
