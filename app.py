import os
import json
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from PIL import Image
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream)
        
        # Use the latest Flash model
        model = genai.GenerativeModel('models/gemini-1.5-flash')

        prompt = """
        Analyze this image's mood/color. Return valid JSON only.
        {
            "vibe_title": "Short creative title (max 5 words)",
            "vibe_description": "One sentence description.",
            "playlist": [
                {"title": "Song", "artist": "Artist"},
                {"title": "Song", "artist": "Artist"},
                {"title": "Song", "artist": "Artist"},
                {"title": "Song", "artist": "Artist"},
                {"title": "Song", "artist": "Artist"}
            ]
        }
        """

        # === FIX: RELAXED SAFETY SETTINGS ===
        # We allow "Medium" and "Low" probability content, blocking only "High" probability threats.
        # This fixes the issue where cars/skin/beaches were getting blocked.
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        try:
            response = model.generate_content([prompt, img], safety_settings=safety_settings)
        except Exception as e:
            print(f"Generation Error: {e}")
            return jsonify({'error': "AI Service Busy. Please try again."}), 500

        # Check if response was blocked
        if not response.parts:
             # Debugging: Print exactly why it was blocked in the terminal
             print(f"Blocked Reason: {response.prompt_feedback}")
             return jsonify({'error': "The AI refused this image. Try a clearer photo."}), 400

        # Clean JSON
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        return jsonify(data)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'error': "Could not process image."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)