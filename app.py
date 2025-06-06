from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="brain_tumor_model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 150, 150, 1)
    return img_array

# Home route (for health check / root access)
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Brain Tumor Detection API is live.",
        "usage": "Send a POST request to /predict with a key 'file' containing an image."
    }), 200

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded. Please include an image with the key 'file'."}), 400

    file = request.files['file']
    try:
        img_array = preprocess_image(file.read())

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Model output interpretation
        if output.shape[-1] == 2:
            prediction = np.argmax(output[0])
            label = "Tumor" if prediction == 1 else "No Tumor"
        else:
            label = "Tumor" if output[0][0] > 0.5 else "No Tumor"

        return jsonify({"result": label}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

# Main block
if __name__ == "__main__":
    app.run(debug=True)
