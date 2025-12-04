# Deployment Guide

This guide explains various ways to deploy the AI-based Skin Disease Detection System.

## 🚀 Deployment Options

### 1. Google Colab (Recommended for Testing)

**Pros:**
- No setup required
- Free GPU access
- Easy to share
- Perfect for demonstrations

**Cons:**
- Session timeouts
- Not suitable for production
- Limited storage

**How to Deploy:**
Simply share the Colab notebook link with users.

---

### 2. Gradio Web App

Create an interactive web interface using Gradio.

**Installation:**
```bash
pip install gradio
```

**Create `app.py`:**
```python
import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
from healthcare_recommendations import format_recommendation

# Load model
model = tf.keras.models.load_model('skin_disease_model.h5')

DISEASE_CLASSES = ['Acne', 'Dermatitis', 'Eczema', 'Melanoma', 
                   'Normal', 'Psoriasis', 'Warts']

def predict_disease(image):
    # Preprocess
    img = Image.fromarray(image).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    disease = DISEASE_CLASSES[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    # Get recommendations
    recommendations = format_recommendation(disease)
    
    return disease, f"{confidence*100:.2f}%", recommendations

# Create interface
iface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(),
    outputs=[
        gr.Textbox(label="Detected Condition"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Healthcare Recommendations")
    ],
    title="AI Skin Disease Detection System",
    description="Upload a skin image to get AI-powered predictions and healthcare recommendations."
)

iface.launch()
```

**Run:**
```bash
python app.py
```

---

### 3. Flask Web Application

Create a full-featured web application with Flask.

**Installation:**
```bash
pip install flask flask-cors
```

**Create `flask_app.py`:**
```python
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from healthcare_recommendations import get_recommendation

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model('skin_disease_model.h5')

DISEASE_CLASSES = ['Acne', 'Dermatitis', 'Eczema', 'Melanoma', 
                   'Normal', 'Psoriasis', 'Warts']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Process image
    img = Image.open(io.BytesIO(file.read())).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    disease = DISEASE_CLASSES[predicted_idx]
    confidence = float(predictions[0][predicted_idx])
    
    # Get recommendations
    recommendations = get_recommendation(disease)
    
    return jsonify({
        'disease': disease,
        'confidence': confidence,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**Run:**
```bash
python flask_app.py
```

---

### 4. Streamlit App

Create an elegant dashboard with Streamlit.

**Installation:**
```bash
pip install streamlit
```

**Create `streamlit_app.py`:**
```python
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from healthcare_recommendations import format_recommendation

# Page config
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🏥",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('skin_disease_model.h5')

model = load_model()

DISEASE_CLASSES = ['Acne', 'Dermatitis', 'Eczema', 'Melanoma', 
                   'Normal', 'Psoriasis', 'Warts']

# UI
st.title("🏥 AI Skin Disease Detection System")
st.write("Upload a skin image to get AI-powered diagnosis and recommendations")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    disease = DISEASE_CLASSES[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    with col2:
        st.subheader("Prediction Results")
        st.metric("Detected Condition", disease)
        st.metric("Confidence", f"{confidence*100:.2f}%")
        
        # Progress bar
        st.progress(float(confidence))
    
    # Recommendations
    st.markdown("---")
    st.subheader("Healthcare Recommendations")
    recommendations = format_recommendation(disease)
    st.text(recommendations)
```

**Run:**
```bash
streamlit run streamlit_app.py
```

---

### 5. Docker Deployment

Create a containerized application.

**Create `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "flask_app.py"]
```

**Build and Run:**
```bash
docker build -t skin-disease-detector .
docker run -p 5000:5000 skin-disease-detector
```

---

### 6. Cloud Platforms

#### Heroku
```bash
# Install Heroku CLI
heroku login
heroku create skin-disease-detector
git push heroku main
```

#### Google Cloud Platform
```bash
gcloud init
gcloud app deploy
```

#### AWS Elastic Beanstalk
```bash
eb init
eb create skin-disease-env
eb deploy
```

---

## 📊 Production Considerations

### 1. Model Optimization
```python
# Convert to TensorFlow Lite for mobile
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 2. API Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per day", "10 per hour"]
)
```

### 3. Security
- Use HTTPS
- Implement authentication
- Validate file uploads
- Sanitize inputs
- Add CORS headers

### 4. Monitoring
- Log predictions
- Track model performance
- Monitor API response times
- Set up alerts

### 5. Scalability
- Use load balancers
- Implement caching
- Use CDN for static files
- Consider serverless options

---

## 🔒 Legal and Compliance

### IMPORTANT: Medical Device Regulations

⚠️ **Before deploying for real medical use:**

1. **FDA Compliance (US)**: May require FDA clearance as a medical device
2. **CE Marking (EU)**: Required for medical devices in Europe
3. **HIPAA Compliance**: If handling patient data
4. **GDPR Compliance**: For European users
5. **Liability Insurance**: Consider professional liability coverage

### Disclaimers Required:
- "This is not a medical device"
- "For educational purposes only"
- "Consult healthcare professionals"
- "Not a substitute for professional diagnosis"

---

## 📈 Performance Optimization

### Model Optimization:
```python
# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
```

### Caching:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def predict_cached(image_hash):
    return model.predict(image)
```

### Batch Processing:
```python
# Process multiple images at once
predictions = model.predict(batch_of_images)
```

---

## 🧪 Testing

### Unit Tests:
```python
import unittest

class TestModel(unittest.TestCase):
    def test_prediction(self):
        result = predict_disease(test_image)
        self.assertIn(result['disease'], DISEASE_CLASSES)
        self.assertGreater(result['confidence'], 0)
        self.assertLess(result['confidence'], 1)
```

### Load Testing:
```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:5000/predict
```

---

## 📚 Additional Resources

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [FastAPI for ML](https://fastapi.tiangolo.com/)
- [MLflow for Model Management](https://mlflow.org/)
- [Kubernetes for Scaling](https://kubernetes.io/)

---

**Choose the deployment option that best fits your use case and requirements!**
