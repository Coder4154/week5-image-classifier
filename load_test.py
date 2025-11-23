from tensorflow import keras

model_path = "model/keras_model.h5"

try:
    model = keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Failed to load model:", e)
