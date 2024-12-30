from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
import requests
from werkzeug.utils import secure_filename
from googletrans import Translator  # Install via `pip install googletrans==4.0.0-rc1`

app = Flask(__name__)

# Disease Descriptions
disease_descriptions = {
    "Apple___Apple_scab": "Apple scab is a fungal disease caused by *Venturia inaequalis*, which affects apple trees, causing dark, scabby lesions on leaves and fruit.",
    "Apple___Black_rot": "Caused by the fungus Botryosphaeria obtusa, it leads to black, sunken lesions on apples and leaves, often accompanied by leaf spotting and twig cankers.",
    "Apple___Cedar_apple_rust": "A fungal disease caused by Gymnosporangium juniperi-virginianae, it creates bright orange, rust-colored lesions on apple leaves and fruits, often requiring both apple and cedar hosts.",
    "Apple___healthy": "Indicates leaves free from diseases or abnormalities, showing optimal health..",
    "Blueberry___healthy": "Represents healthy blueberry plants with no signs of disease or stress.",
    "Cherry_(including_sour)___Powdery_mildew": "A fungal disease caused by Podosphaera clandestina, forming white, powdery spots on leaves, stems, and fruits, reducing photosynthesis and fruit quality.",
    "Cherry_(including_sour)___healthy": "Indicates healthy cherry plants without visible disease symptoms.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "A fungal disease caused by Cercospora zeae-maydis, leading to rectangular, gray or tan lesions on leaves, reducing photosynthetic capacity and yield.",
    "Corn_(maize)___Common_rust":"Caused by Puccinia sorghi, it creates reddish-brown pustules on leaves and stalks, potentially impacting yield if severe.",
    "Corn_(maize)___Northern_Leaf_Blight": "Caused by Setosphaeria turcica, it forms elongated, cigar-shaped lesions on leaves, which can merge to cause widespread blight.",
    "Corn_(maize)___healthy" : "Indicates corn leaves free from diseases or damage.",
    "Grape___Black_rot": "Caused by the fungus Guignardia bidwellii, it forms small, dark spots on leaves and black, shriveled areas on fruits, leading to rot.",
    "Grape___Esca_(Black_Measles)": "A fungal disease complex caused by several fungi, leading to dark streaks on wood, leaf discoloration, and black spots on fruits.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Caused by Isariopsis clavispora, it creates dark brown lesions on leaves, reducing photosynthesis.",
    "Grape___healthy": "Represents healthy grape plants with no disease symptoms.",
    "Peach___Bacterial_spot": "Caused by Xanthomonas campestris pv. pruni, it creates small, dark spots on leaves, fruits, and twigs, reducing fruit quality and yield.",
    "Peach___healthy": "Represents peach plants without disease symptoms.",
    "Pepper,_bell___Bacterial_spot": "Caused by Xanthomonas campestris pv. vesicatoria, leading to small, water-soaked lesions on leaves and fruits that turn dark and scabby.",
    "Pepper,_bell___healthy": "Indicates healthy bell pepper plants without disease.",
    "Potato___Early_blight": "A fungal disease caused by Alternaria solani, forming concentric ring patterns on leaves, leading to premature defoliation.",
    "Potato___Late_blight": "Caused by Phytophthora infestans, it produces water-soaked lesions on leaves and stems that rapidly turn brown and spread, causing major crop loss.",
    "Potato___healthy": "Represents healthy potato plants free from disease.",
    "Raspberry___healthy": "Indicates raspberry plants free from visible diseases or stress.",
    "Soybean___healthy": "Represents soybean plants without disease symptoms.",
    "Squash___Powdery_mildew": "A fungal disease caused by Erysiphe cichoracearum, forming white, powdery spots on leaves and stems, reducing photosynthesis.",
    "Strawberry___Leaf_scorch": "Caused by the fungus Diplocarpon earlianum, it forms dark spots on leaves that can merge to create scorched or dead areas.",
    "Strawberry___healthy": "Indicates strawberry plants without visible disease symptoms.",
    "Tomato___Bacterial_spot": "Caused by Xanthomonas campestris pv. vesicatoria, it creates small, dark lesions on leaves, stems, and fruits.",
    "Tomato___Early_blight": "A fungal disease caused by Alternaria solani, forming concentric, brown lesions on leaves, stems, and fruits.",
    "Tomato___Late_blight": "Caused by Phytophthora infestans, leading to water-soaked, rapidly spreading lesions on leaves, stems, and fruits.",
    "Tomato___Leaf_Mold": "Caused by Passalora fulva, creating yellow spots on the upper surface and mold growth underneath.",
    "Tomato___Septoria_leaf_spot": "A fungal disease caused by Septoria lycopersici, forming small, circular spots on leaves, leading to defoliation.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Infestation by Tetranychus urticae, causing yellowing and webbing on leaves.",
    "Tomato___Target_Spot": "Caused by Corynespora cassiicola, forming circular, target-like spots on leaves and fruits.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "A viral disease spread by whiteflies, causing yellowing, curling leaves, and stunted growth.",
    "Tomato___Tomato_mosaic_virus": "A viral disease causing mottled, distorted leaves and reduced fruit yield.",
    "Tomato___healthy":"Represents healthy tomato plants without disease or stress.",
}

# Function to handle predictions
def model_prediction(test_image_path):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element



# Translator function
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Route: Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Route: About Page
@app.route("/about")
def about():
    return render_template("about.html")

# Route: Disease Recognition Page
@app.route("/disease_recognition", methods=["GET", "POST"])
def disease_recognition():
    prediction_result = None
    disease_description = None
    translated_description = None

    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join("uploads", filename)
            file.save(filepath)

            # Perform prediction
            result_index = model_prediction(filepath)

            # Reading Labels
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
           
            prediction_result = class_names[result_index]
            disease_description = disease_descriptions.get(prediction_result, "No description available for this disease.")

            # Translate the description only
            target_language = request.form.get("language")
            if target_language:
                translated_description = translate_text(disease_description, target_language)

    return render_template(
        "disease_recognition.html",
        prediction_result=prediction_result,
        disease_description=disease_description,
        translated_description=translated_description
    )




if __name__ == "__main__":
    app.run(debug=True)
