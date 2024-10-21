from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = load_model('denoising_autoencoder.h5')

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L") 

    # Resize the image to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0 

    # Add batch dimension and channel dimension
    image_array = np.expand_dims(image_array, axis=0)  
    image_array = np.expand_dims(image_array, axis=-1) 

    return image_array

def denoise_image(image_array):
    denoised_image_array = model.predict(image_array)
    return denoised_image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    # Save the uploaded image
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    # Load and preprocess the image
    image = Image.open(img_path)
    processed_image = preprocess_image(image)

    # Denoise the image
    denoised_image_array = denoise_image(processed_image)

    # Print the shape of the denoised image array for debugging
    print("Denoised image array shape:", denoised_image_array.shape)

    if denoised_image_array.ndim == 2:
        denoised_image = Image.fromarray((denoised_image_array * 255).astype(np.uint8), mode='L')
    elif denoised_image_array.ndim == 3:
        # Assuming the last dimension is for channels
        denoised_image = Image.fromarray((denoised_image_array * 255).astype(np.uint8))
    else:
        return 'Invalid output shape from model', 500

    # Save the denoised image
    denoised_path = os.path.join('static', 'denoised_' + file.filename)
    denoised_image.save(denoised_path)

    return render_template('result.html', original=img_path, denoised=denoised_path)

if __name__ == '__main__':
    app.run(debug=True)
