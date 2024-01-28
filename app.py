from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = load_model('best_model.h5')

def predict_plant(file):
    img = Image.open(BytesIO(file.read()))
    img = img.resize((256, 256))  # Resize the image to match the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    # Modify this part based on your model's class labels
    class_labels = ["almond", "banana", "cardamom", "Cherry", "chilli", "clove", "coconut",
                    "Coffe-plant", "cotton", "Cucumber", "Fox_nut", "gram", "jowar", "jute",
                    "Lemon", "maize", "mustard-oil", "Olive-tree", "papaya", "Pearl_millet",
                    "pineapple", "rice", "soyabean", "sugarcane", "sunflower", "tea", "Tobacco-plant",
                    "tomato", "vigna-radiati", "wheat"] 
    plant_name = class_labels[predicted_class-1]

    return plant_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    plant_name = predict_plant(file)
    return jsonify({'plant_name': plant_name})

if __name__ == '__main__':
    app.run(debug=True)
