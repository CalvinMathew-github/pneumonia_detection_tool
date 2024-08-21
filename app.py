from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import numpy as np
import mysql.connector
import os
from urllib.parse import quote, unquote

import uuid
from PIL import Image
import io
from werkzeug.utils import secure_filename
from preprocess_image import preprocess_img
from model_predict  import pred_disease
disease_dic = ["Bacterial_Pneumonia", "Normal", "Viral_Pneumonia", "unknown"]
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configurations
db_config = {
    'user': 'root',
    'password': 'CAl@1234',
    'host': 'localhost',
    'database': 'myserver'
}

def get_db_connection():
    conn = mysql.connector.connect(**db_config)
    return conn

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_query = request.form['search_query']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM person_name WHERE given_name LIKE %s", ('%' + search_query + '%',))
        patients = cursor.fetchall()
        conn.close()
        return render_template('search_results.html', patients=patients)
    return render_template('search.html')

@app.route('/home')
def home():
    title = 'Pneumonia Detection and Classification'
    return render_template('index.html', title=title)

@app.route('/check-obs-for-patient/<int:patient_id>', methods=['GET'])
def check_obs_for_patient(patient_id):
    title = 'Observations'
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM obs WHERE person_id = %s AND value_text IS NOT NULL AND voided = 0", (patient_id,))
    observations = cursor.fetchall()
    conn.close()

    # Debugging print statements
    print(f"Patient ID: {patient_id}")
    print(f"Observations: {observations}")

    
    return render_template('obs_results.html', observations=observations, title=title)

@app.route('/view-image/<path:image_path>')
def view_image(image_path):
    try:
        # Ensure the image_path is secure
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        flash(f"Error loading image: {e}")
        return redirect(url_for('index'))


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Pneumonia Classification'

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        
        img1 = file.read()
        with open('output1.png', 'wb') as output_file:
            output_file.write(img1)

        try:
            preprocess_img()
        except Exception as e:
            return render_template('error_page.html', prediction="Error", precaution="Try to Upload Image Before Predicting.", title=title)

        index_result, densenet_scores, efficientnet_scores, ensembled_scores = pred_disease("output_hsv.png")
        predicted_class = disease_dic[index_result]

        if predicted_class == "Bacterial_Pneumonia":
         precaution = "Condition is Bacterial Pneumonia"
        elif predicted_class == "Normal":
         precaution = "Condition is Normal"
        elif predicted_class == "Viral_Pneumonia":
         precaution = "Condition is Viral Pneumonia"
        else:
         precaution = "Please Upload X-ray Images. Unknown image."

        return render_template('disease-result.html', 
                               prediction=predicted_class, 
                               precaution=precaution, 
                               title=title,
                               densenet_scores=densenet_scores, 
                               efficientnet_scores=efficientnet_scores, 
                               ensembled_scores=ensembled_scores)
    return render_template('disease.html', title=title)

@app.route('/predict_disease/<path:image_path>', methods=['GET'])
def predict_disease(image_path):
    title = 'Pneumonia Classification'
    file_path = image_path
     
    try:
        img = Image.open(file_path)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        with open('output1.png', 'wb') as output_file:
            output_file.write(img_byte_arr)
        
        preprocess_img()
        index_result, densenet_scores, efficientnet_scores, ensembled_scores = pred_disease("output_hsv.png")
        predicted_class = disease_dic[index_result]

        if predicted_class == "Bacterial_Pneumonia":
         precaution = "Condition is Bacterial Pneumonia"
        elif predicted_class == "Normal":
         precaution = "Condition is Normal"
        elif predicted_class == "Viral_Pneumonia":
         precaution = "Condition is Viral Pneumonia"
        else:
         precaution = "Please Upload X-ray Images. Unknown image."

        return render_template('disease-result.html', 
                               prediction=predicted_class, 
                               precaution=precaution, 
                               title=title,
                               densenet_scores=densenet_scores, 
                               efficientnet_scores=efficientnet_scores, 
                               ensembled_scores=ensembled_scores)
    except Exception as e:
        flash(f"Error in processing image: {e}")
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
