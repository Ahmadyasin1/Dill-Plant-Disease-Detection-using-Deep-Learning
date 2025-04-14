import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to filter out all logs except ERROR
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')


# Define route for serving the index.html file
@app.route('/')
def index():
    return render_template('index.html')


# Define a route for predicting plant diseases
@app.route('/predict', methods=['POST'])
def result():
    # Receive the uploaded image
    file = request.files['file']
    # Save the image to a temporary location
    image_path = 'temp.jpg'
    file.save(image_path)

    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    # Make predictions
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)
    class_name = ['bacterial blight', 'healthy', 'leaf spot', 'powdery mildew']  # Update with your class names
    print(result_index)
    # Get the predicted class name
    model_prediction = class_name[result_index-1]

    # Return the prediction result
    return render_template('result.html', prediction=model_prediction)


if __name__ == '__main__':
    app.run(debug=True)
