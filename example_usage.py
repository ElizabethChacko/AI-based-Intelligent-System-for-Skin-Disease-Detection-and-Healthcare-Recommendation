"""
Example Usage Script for Skin Disease Detection System

This script demonstrates how to use the trained model programmatically
for predictions without using the Colab notebook interface.

Prerequisites:
1. Train the model using the Colab notebook
2. Download the saved model file (skin_disease_model.h5)
3. Install required packages: pip install -r requirements.txt

Author: Elizabeth Chacko
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os
from healthcare_recommendations import format_recommendation, is_emergency


# Configuration
# Note: This should match the filename saved in the Colab notebook
# Default is 'skin_disease_detection_model.h5' as saved by the notebook
MODEL_PATH = 'skin_disease_detection_model.h5'
IMG_HEIGHT = 224
IMG_WIDTH = 224

DISEASE_CLASSES = [
    'Acne',
    'Dermatitis',
    'Eczema',
    'Melanoma',
    'Normal',
    'Psoriasis',
    'Warts'
]


def load_model(model_path=MODEL_PATH):
    """
    Load the trained model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please train the model using the Colab notebook first.")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    return model


def preprocess_image(image_path):
    """
    Preprocess an image for prediction.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Preprocessed image array and original image
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at {image_path}")
        return None, None
    
    try:
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        original_img = img.copy()
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, original_img
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None, None


def predict_disease(model, image_path, verbose=True):
    """
    Predict skin disease from an image.
    
    Args:
        model: Trained Keras model
        image_path (str): Path to the image file
        verbose (bool): Whether to print detailed output
        
    Returns:
        Dictionary containing prediction results
    """
    # Preprocess image
    img_array, original_img = preprocess_image(image_path)
    
    if img_array is None:
        return None
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = DISEASE_CLASSES[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        (DISEASE_CLASSES[idx], predictions[0][idx])
        for idx in top_3_idx
    ]
    
    results = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top_3_predictions': top_3_predictions,
        'all_predictions': predictions[0],
        'is_emergency': is_emergency(predicted_class)
    }
    
    if verbose:
        print_results(results, image_path)
    
    return results


def print_results(results, image_path):
    """
    Print prediction results in a formatted way.
    
    Args:
        results (dict): Prediction results
        image_path (str): Path to the analyzed image
    """
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    print(f"\n📁 Image: {image_path}")
    print(f"🔍 Detected Condition: {results['predicted_class']}")
    print(f"📊 Confidence: {results['confidence']*100:.2f}%")
    
    if results['is_emergency']:
        print("⚠️  WARNING: This condition may require IMMEDIATE medical attention!")
    
    print("\n📋 Top 3 Predictions:")
    for i, (disease, conf) in enumerate(results['top_3_predictions'], 1):
        print(f"   {i}. {disease}: {conf*100:.2f}%")
    
    print("\n" + "="*80)
    print("HEALTHCARE RECOMMENDATIONS")
    print("="*80)
    print(format_recommendation(results['predicted_class']))


def batch_predict(model, image_paths):
    """
    Predict diseases for multiple images.
    
    Args:
        model: Trained Keras model
        image_paths (list): List of image file paths
        
    Returns:
        List of prediction results
    """
    results = []
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Analyzing: {image_path}")
        result = predict_disease(model, image_path, verbose=False)
        
        if result:
            results.append({
                'image_path': image_path,
                'prediction': result
            })
            print(f"  Result: {result['predicted_class']} ({result['confidence']*100:.2f}%)")
        else:
            print(f"  ❌ Failed to process image")
    
    return results


def main():
    """
    Main function demonstrating example usage.
    """
    print("="*80)
    print("SKIN DISEASE DETECTION SYSTEM - EXAMPLE USAGE")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model file not found: {MODEL_PATH}")
        print("\nTo use this script:")
        print("1. Open and run Skin_Disease_Detection_System.ipynb in Google Colab")
        print("2. Train the model (it will be saved automatically)")
        print("3. Download the saved model file: skin_disease_model.h5")
        print("4. Place it in the same directory as this script")
        print("5. Run this script again")
        return
    
    # Load model
    model = load_model()
    
    # Example 1: Single image prediction
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*80)
    
    # Check if example image exists
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nAnalyzing image from command line: {image_path}")
        predict_disease(model, image_path)
    else:
        print("\nNo image provided.")
        print("\nUsage:")
        print("  python example_usage.py <path_to_image>")
        print("\nExample:")
        print("  python example_usage.py /path/to/skin_image.jpg")
    
    # Example 2: Batch prediction (if multiple arguments provided)
    if len(sys.argv) > 2:
        print("\n" + "="*80)
        print("EXAMPLE 2: Batch Prediction")
        print("="*80)
        
        image_paths = sys.argv[1:]
        results = batch_predict(model, image_paths)
        
        print("\n" + "="*80)
        print("BATCH PREDICTION SUMMARY")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            pred = result['prediction']
            print(f"\n{i}. {result['image_path']}")
            print(f"   Prediction: {pred['predicted_class']}")
            print(f"   Confidence: {pred['confidence']*100:.2f}%")
            if pred['is_emergency']:
                print(f"   ⚠️  URGENT: Requires immediate medical attention")
    
    print("\n" + "="*80)
    print("DISCLAIMER")
    print("="*80)
    print("⚠️  This is an AI-based prediction system for educational purposes.")
    print("   Always consult qualified healthcare professionals for accurate")
    print("   diagnosis and treatment. Do not rely solely on this system.")
    print("="*80)


if __name__ == "__main__":
    main()
