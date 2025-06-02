from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load your trained models
alzheimers_model = tf.keras.models.load_model('models/alzheimer_model.keras')
brainstroke_model = tf.keras.models.load_model('models/brainstroke_model.keras')
brain_model = tf.keras.models.load_model('models/brain_model.keras')

def preprocess_image(image):
    # Preprocess the image for your model (resize, normalize, etc.)
    image = image.resize((224, 224))  # Example size
    image_array = np.array(image) / 255.0  # Normalize
    return image_array.reshape(1, 224, 224, 3)  # Reshape for model input

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    disease = data['disease']
    image_file = request.files['image']

    # Preprocess the image
    image = Image.open(image_file)
    processed_image = preprocess_image(image)

    # Make predictions based on the selected disease
    if disease == 'Alzheimer\'s':
        prediction = alzheimers_model.predict(processed_image)
    elif disease == 'Brain Stroke':
        prediction = brainstroke_model.predict(processed_image)
    elif disease == 'Brain Tumor':
        prediction = brain_model.predict(processed_image)
    else:
        return jsonify({'error': 'Invalid disease selected'}), 400

    # Return the prediction result
    result = 'Disease Present' if prediction[0][0] >= 0.5 else 'No Disease Detected'
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)