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
        
        # Use the fast Flash model
        model = genai.GenerativeModel('models/gemini-flash-latest')

        # Optimized Prompt for Speed
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

        # SECURITY SYSTEM: Strict blocking for harmful content
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }

        try:
            response = model.generate_content([prompt, img], safety_settings=safety_settings)
        except Exception:
            # If the API refuses to generate, it's usually a safety block
            return jsonify({'error': "SECURITY ALERT: This image violates safety guidelines (Violence/Nudity/Hate). Request blocked."}), 400

        # Check if response was blocked by safety filters
        if not response.parts:
             return jsonify({'error': "SECURITY ALERT: The AI flagged this image as unsafe."}), 400

        # Clean JSON
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        return jsonify(data)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': "Could not read the vibe. Try a different photo."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)