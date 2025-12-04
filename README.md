# AI-Based Intelligent System for Skin Disease Detection and Healthcare Recommendation

An advanced deep learning-based system for automated skin disease detection and personalized healthcare recommendations, designed to run seamlessly in Google Colab.

## 🌟 Features

- **🔬 Automated Disease Detection**: Uses state-of-the-art deep learning (CNN with MobileNetV2) to classify 7 different skin conditions
- **💊 Healthcare Recommendations**: Provides personalized treatment suggestions and care guidelines for each detected condition
- **📊 High Accuracy**: Transfer learning approach for robust and accurate predictions
- **🖼️ Easy Image Upload**: User-friendly interface for uploading and analyzing skin images
- **📈 Visual Results**: Clear visualization of predictions with confidence scores
- **☁️ Cloud-Ready**: Optimized for Google Colab with GPU acceleration support

## 🩺 Supported Skin Conditions

1. **Acne** - Common inflammatory skin condition
2. **Dermatitis** - Skin inflammation and irritation
3. **Eczema** - Chronic inflammatory skin condition
4. **Melanoma** - Serious form of skin cancer (requires immediate medical attention)
5. **Normal Skin** - Healthy skin with no detected conditions
6. **Psoriasis** - Autoimmune skin condition
7. **Warts** - Viral skin infections

## 🚀 Getting Started

### Option 1: Run in Google Colab (Recommended)

1. Open the notebook in Google Colab:
   - Click on `Skin_Disease_Detection_System.ipynb`
   - Click the "Open in Colab" badge (or upload to Colab)

2. Run all cells sequentially:
   - Click `Runtime` → `Run all`
   - Or press `Ctrl+F9` (Windows/Linux) or `Cmd+F9` (Mac)

3. Upload your skin image when prompted and get instant predictions!

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/ElizabethChacko/AI-based-Intelligent-System-for-Skin-Disease-Detection-and-Healthcare-Recommendation.git

# Navigate to the directory
cd AI-based-Intelligent-System-for-Skin-Disease-Detection-and-Healthcare-Recommendation

# Install dependencies
pip install -r requirements.txt

# Open the notebook in Jupyter
jupyter notebook Skin_Disease_Detection_System.ipynb
```

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- PIL/Pillow
- Google Colab (for cloud execution)

See `requirements.txt` for complete list of dependencies.

## 🏗️ System Architecture

### Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned for skin disease classification
- **Input Size**: 224x224 RGB images
- **Output**: 7-class classification with confidence scores

### Pipeline Overview
```
Input Image → Preprocessing → CNN Model → Classification → Healthcare Recommendations
```

### Key Components:
1. **Data Preprocessing**: Image resizing, normalization, and augmentation
2. **Model Training**: Transfer learning with MobileNetV2
3. **Prediction Engine**: Real-time disease classification
4. **Recommendation System**: Context-aware healthcare advice

## 📊 Model Performance

The model uses transfer learning with the following architecture:
- **MobileNetV2** as the base model (frozen layers)
- **Global Average Pooling** layer
- **Dense layers** with dropout for regularization
- **Softmax activation** for multi-class classification

Training includes:
- Data augmentation for improved generalization
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Batch normalization for stable training

## 💡 Usage Example

```python
# Upload an image
from google.colab import files
uploaded = files.upload()

# Make prediction
results = predict_disease(model, uploaded_image)

# Display results with recommendations
display_prediction_results(results)
```

## 🎯 Healthcare Recommendations

For each detected condition, the system provides:
- **Description**: Detailed explanation of the condition
- **Severity Level**: Risk assessment
- **Treatment Recommendations**: Step-by-step care instructions
- **When to See a Doctor**: Guidelines for seeking professional help

## ⚠️ Important Disclaimer

**This system is for educational and research purposes only.**

- ❌ Do NOT use as the sole basis for medical decisions
- ❌ Do NOT replace professional medical consultation
- ✅ Always consult qualified healthcare professionals
- ✅ Use as a supplementary tool for awareness
- ✅ Seek immediate medical attention for serious conditions

## 🔧 Customization

### Using Your Own Dataset

Replace the synthetic data generation with real datasets:

```python
# Example: Load HAM10000 dataset
# Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

train_dir = 'path/to/ham10000/train'
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
```

### Fine-Tuning the Model

```python
# Unfreeze more layers for fine-tuning
base_model.trainable = True

# Compile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
model.fit(train_generator, epochs=10)
```

## 📚 Recommended Datasets

For production use, consider these datasets:

1. **HAM10000**: 10,000+ dermatoscopic images
   - [Kaggle Link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

2. **ISIC Archive**: International Skin Imaging Collaboration
   - [Website](https://www.isic-archive.com/)

3. **DermNet**: Comprehensive dermatology image database
   - [Website](https://dermnetnz.org/)

## 🛠️ Advanced Features

The notebook includes optional advanced features:
- **Confusion Matrix**: Visualize model performance
- **Fine-Tuning**: Improve model accuracy
- **Classification Report**: Detailed metrics per class
- **Grad-CAM**: Visualize what the model is looking at (can be added)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Adding more disease classes
- Implementing advanced architectures
- Adding explainable AI features (Grad-CAM)
- Creating deployment scripts
- Improving UI/UX

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Elizabeth Chacko** - Initial work

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the framework
- MobileNetV2 architecture developers
- Medical professionals for domain knowledge
- Open-source skin disease datasets

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review the notebook comments

## 🔗 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Skin Cancer Detection Research](https://www.nature.com/articles/nature21056)

---

**⭐ If you find this project helpful, please give it a star!**

**Made with ❤️ for advancing healthcare through AI**