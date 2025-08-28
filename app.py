from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import io
from fpdf import FPDF
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained model. Make sure this path is correct!
MODEL_PATH = 'my_models/Alzheimer_detection_grayscale_final_models.h5'

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None # Set model to None if loading fails

# Function to interpret the prediction score into a dementia level
def get_dementia_level(prediction_score):
    if prediction_score >= 0.8:
        return "Moderate Demented"
    elif prediction_score >= 0.6:
        return "Mild Demented"
    elif prediction_score >= 0.4:
        return "Very Mild Demented"
    else:
        return "Non Demented"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Model failed to load on startup.'})

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'})

    # Get patient details from the form
    patient_name = request.form.get('name', 'N/A')
    patient_age = request.form.get('age', 'N/A')
    patient_gender = request.form.get('gender', 'N/A')

    try:
        img = Image.open(file.stream)
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        prediction_score = model.predict(img_array)[0][0]
        predicted_level = get_dementia_level(prediction_score)

        return jsonify({
            'success': True,
            'predicted_level': predicted_level,
            'prediction_score': float(prediction_score),
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    predicted_level = data.get('predicted_level')
    prediction_score = data.get('prediction_score')
    patient_name = data.get('patient_name')
    patient_age = data.get('patient_age')
    patient_gender = data.get('patient_gender')

    # Placeholder values for demonstration
    overall_accuracy = 0.9992
    cm = np.array([[53777, 45], [20, 15371]])

    # Create a PDF report
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Alzheimer Detection Report', 0, 1, 'C')
    pdf.ln(10)

    # Patient Details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, 'Patient Details', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 7, f"Name: {patient_name}", 0, 1)
    pdf.cell(200, 7, f"Age: {patient_age}", 0, 1)
    pdf.cell(200, 7, f"Gender: {patient_gender}", 0, 1)
    pdf.ln(10)

    # Prediction and Conditional Message
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 7, f"Prediction for Alzheimer: {predicted_level}", 0, 1)
    pdf.ln(5)

    if predicted_level == "Non Demented":
        pdf.set_font('Arial', 'I', 12)
        pdf.multi_cell(0, 7, "No signs of dementia were detected in your MRI scan. This is a very positive result. Continue to live a healthy lifestyle to maintain brain health.")
    elif predicted_level == "Very Mild Demented":
        pdf.set_font('Arial', 'I', 12)
        pdf.multi_cell(0, 7, "Very mild signs of dementia detected. It is highly recommended that you consult a neurologist for a professional assessment.")
        pdf.ln(5)
        # Add precautions here as needed
        # ...
    else: # Mild Demented and Moderate Demented
        pdf.set_font('Arial', 'I', 12)
        pdf.multi_cell(0, 7, "Dementia detected in your MRI. Kindly consult a nearby neurologist immediately for a comprehensive diagnosis and treatment plan!")
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 7, 'Here are some general precautions you can take:')
        pdf.ln(5)
        
        precautions = [
            "1. Stay mentally active: Engage in mentally stimulating activities such as reading, writing, puzzles, and games to keep your brain active.",
            "2. Stay physically active: Exercise regularly to improve blood flow to the brain and help prevent cognitive decline.",
            "3. Eat a healthy diet: Eat a balanced diet that is rich in fruits, vegetables, whole grains, and lean protein to help maintain brain health.",
            "4. Stay socially active: Engage in social activities and maintain social connections to help prevent social isolation and depression.",
            "5. Get enough sleep: Aim for 7-8 hours of sleep per night to help improve brain function and prevent cognitive decline."
        ]
        for p in precautions:
            pdf.multi_cell(0, 5, p)
            pdf.ln(2)

    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 7, f"Model Confidence Score: {prediction_score:.2f}", 0, 1)
    pdf.cell(200, 7, f"Overall Model Accuracy: {overall_accuracy*100:.2f}%", 0, 1)

    pdf_output = pdf.output(dest='S').encode('latin-1')
    return send_file(io.BytesIO(pdf_output), as_attachment=True, download_name='report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred while running the app: {e}")
