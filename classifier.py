# classifier.py

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class ImageClassifier:
    """
    Image classification using a Teachable Machine model
    """

    def __init__(self, model_path='model/keras_model.h5', labels_path='model/labels.txt'):
        print("Loading model...")
        self.model = keras.models.load_model(model_path, compile=False)
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        print(f"Model loaded with {len(self.labels)} classes: {self.labels}")
        # Model expects 224x224 images
        self.image_size = (224, 224)

    def preprocess_image(self, image_path):
        """Open, resize, normalize, and add batch dimension to an image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size)
        img_array = np.array(img) / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)  # add batch
        return img_array, img

    def classify_image(self, image_path):
        """Predict classes and confidence for a single image"""
        processed_image, original_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image, verbose=0)

        results = []
        for i, label in enumerate(self.labels):
            confidence = float(predictions[0][i]) * 100
            results.append({'class': label, 'confidence': confidence})

        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results, original_image

    def visualize_prediction(self, image_path):
        """Show image and top predictions as bar chart"""
        results, img = self.classify_image(image_path)

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Input Image')

        classes = [r['class'] for r in results[:3]]
        confidences = [r['confidence'] for r in results[:3]]
        ax2.barh(classes, confidences)
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title('Top 3 Predictions')
        ax2.set_xlim(0, 100)

        for i, (c, conf) in enumerate(zip(classes, confidences)):
            ax2.text(conf + 1, i, f'{conf:.1f}%')

        plt.tight_layout()
        plt.savefig('prediction_result.png')
        plt.show()

        print("\n🎯 Classification Results:")
        print("-" * 40)
        for r in results:
            print(f"{r['class']:15} {r['confidence']:6.2f}%")

        return results[0]  # return top prediction

    def classify_from_webcam(self):
        """Realtime webcam classification"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert frame to PIL image and preprocess
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(self.image_size)
            img_array = np.expand_dims(np.array(pil_img)/255.0, axis=0)

            predictions = self.model.predict(img_array, verbose=0)
            top_idx = np.argmax(predictions)
            label = self.labels[top_idx]
            confidence = predictions[0][top_idx] * 100

            # Display label on frame
            cv2.putText(frame, f"{label} ({confidence:.1f}%)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Webcam Classifier", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
