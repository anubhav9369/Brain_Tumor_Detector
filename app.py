from flask import Flask, request, jsonify
from flask_cors import CORS  # ← Add this
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # ← Enable CORS for all routes

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="/Users/anubhavverma/Desktop/BrainTumorDetector/brain_tumor_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 150, 150, 1)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img_array = preprocess_image(file.read())

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print("Raw output from model:", output)

    # Assuming model output shape is (1, 2) -> [No Tumor prob, Tumor prob]
    if output.shape[-1] == 2:
        prediction = np.argmax(output[0])
        label = "Tumor" if prediction == 1 else "No Tumor"
    else:
        label = "Tumor" if output[0][0] > 0.5 else "No Tumor"

    return jsonify({"result": label})

if __name__ == "__main__":
    app.run(debug=True)
