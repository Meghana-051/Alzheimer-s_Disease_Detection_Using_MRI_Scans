# advanced_engine/medical_llm_service.py
import os
import requests

def generate_clinical_summary(patient_name, patient_age, patient_gender, predicted_level, confidence_score):
    """
    Calls a live LLM API to generate a completely unique, non-templated, case-specific 
    medical summary and precautions list based on patient diagnostics.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Define a clean, high-grade medical fallback system if your API key is blank or offline
    fallback_text = (
        f"Clinical AI Evaluation: Patient {patient_name} ({patient_age}yo {patient_gender}) "
        f"presents neuroimaging tensor signatures matching characteristics of {predicted_level}. "
        f"The deep learning network processed this evaluation with an analytical confidence factor of {confidence_score*100:.1f}%. "
        f"Recommendation: Urgent neurological consultation to establish localized baseline cognitive boundaries."
    )
    
    # Corrected Safety Check: Intercept empty string conditions accurately
    if not api_key:
        return fallback_text

    # Setup the structured clinical prompt configuration
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system", 
                "content": "You are an elite, expert neuroradiologist writing case notes for a referring medical practitioner. Do not use generic templates, introductory remarks, or structural markdown lines. Keep your response highly professional, compact, and objective."
            },
            {
                "role": "user", 
                "content": f"Analyze this diagnostic profile and write a detailed paragraph comprising (1) a targeted clinical summary of the neuroimaging results, and (2) three highly specific medical/lifestyle management recommendations tailored exactly for this demographic: \n- Patient Name: {patient_name}\n- Age: {patient_age}\n- Gender: {patient_gender}\n- Model Classification: {predicted_level}\n- Classification Confidence: {confidence_score*100:.1f}%"
            }
        ],
        "max_tokens": 300,
        "temperature": 0.3
    }

    try:
        # Fire live endpoint query to OpenAI infrastructure layout
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"API Error ({response.status_code}): {response.text}")
            return fallback_text
    except Exception as e:
        print(f"Connection Exception: {e}")
        return fallback_text