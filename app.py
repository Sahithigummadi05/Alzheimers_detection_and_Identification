from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model('alzheimer_cnn_model.h5')
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    
    file = request.files['file']
    img = Image.open(file).resize((128, 128)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    conf = float(preds[0][idx] * 100)
    
    return jsonify({
        'prediction': CLASSES[idx],
        'confidence': f"{conf:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)