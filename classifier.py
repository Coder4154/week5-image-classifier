# classifier.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class ImageClassifier:
    """
    Image classification using computer vision
    Demonstrates concepts from Chapter 24
    """
    def __init__(self, model_path='model/keras_model.h5', labels_path='model/labels.txt'):
        """Load the trained model and labels"""
        print("Loading model...")
        # Compile=False is required for loading models exported from Teachable Machine
        self.model = keras.models.load_model(model_path, compile=False) 
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        print(f"Model loaded with {len(self.labels)} classes: {self.labels}")
        
        # Model expects 224x224 images
        self.image_size = (224, 224)

    def preprocess_image(self, image_path):
        """
        Prepare image for classification
        This is like the preprocessing we did for text!
        """
        # Open and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size)
        
        # Convert to array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0 # Normalize pixel values
        
        # Add batch dimension (model expects a batch of images, even if it's just one)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img

    def classify_image(self, image_path):
        """Classify a single image"""
        # Preprocess
        processed_image, original_image = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get results
        results = []
        # predictions[0] is the list of confidence scores for the single image
        for i, label in enumerate(self.labels):
            # Scale confidence to percentage
            confidence = float(predictions[0][i]) * 100 
            results.append({
                'class': label,
                'confidence': confidence
            })
            
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results, original_image

    def visualize_prediction(self, image_path):
        """Show image with prediction results"""
        results, img = self.classify_image(image_path)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        ax1.imshow(img)
        ax1.axis('off')
        
        # Create Bar Chart for results
        classes = [r['class'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Highlight the top prediction in a different color
        bar_colors = ['blue'] * len(classes)
        bar_colors[0] = 'green' 
        
        ax2.barh(classes, confidences, color=bar_colors)
        ax2.set_xlim(0, 100) # Confidence is 0-100%
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title('Classification Results')
        
        # Add a text label for the overall prediction
        top_prediction = results[0]
        fig.suptitle(f"Prediction: {top_prediction['class']} ({top_prediction['confidence']:.2f}%)", fontsize=16, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        plt.show()

# Example usage (will not run until model files are in place)
if __name__ == '__main__':
    try:
        # NOTE: Replace 'test_images/your_image.jpg' with a real image path once you have one
        test_image_path = 'test_images/sample.jpg' 
        
        # The file will only load successfully *after* you complete Part 1
        classifier = ImageClassifier() 
        print(f"\nClassifying: {test_image_path}")
        
        # To see the plot, uncomment the line below after you've installed all dependencies
        # classifier.visualize_prediction(test_image_path) 
        
        # To see raw results
        results, _ = classifier.classify_image(test_image_path)
        for r in results:
            print(f"- {r['class']}: {r['confidence']:.2f}%")
            
    except FileNotFoundError as e:
        print("\n--- ERROR ---")
        print("Model files (keras_model.h5 or labels.txt) not found in the 'model' folder.")
        print("Please complete Part 1 on Teachable Machine and place the files in 'model/'.")
        print(f"Also ensure you have a test image at the path: {test_image_path}")
        print("-------------")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")