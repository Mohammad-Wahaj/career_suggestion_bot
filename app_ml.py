from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import json
from ml_model import CareerAdviceMLModel
import os
from openai import OpenAI

app = Flask(__name__)
API_KEY_URL = "https://get-key-isxn.onrender.com/get_key"
CORS(app)

# Initialize the ML model
print("Loading AI/ML Career Advice Model...")
ml_model = CareerAdviceMLModel()

# Try to load existing model, if not found, train a new one
if os.path.exists("models/vectorizer.pkl"):
    ml_model.load_model()
else:
    print("No existing model found. Training new model...")
    ml_model.train_model("dataset/career_dataset.json")

print("AI/ML Career Advice Model loaded successfully!")

def get_openai_client():
    try:
        # API call to get the key
        response = requests.get(API_KEY_URL)
        data = response.json()
        api_key = data.get("api_key")

        if not api_key:
            raise ValueError("API key not found!")

        # Initialize OpenAI client with fetched key
        return OpenAI(api_key=api_key)

    except Exception as e:
        print("Error fetching API key:", e)
        return None

# @app.route("/chat", methods=["POST"])
# def chat():
#     # user_input JSON se lo
#     user_message = request.json.get("user_input", "")

#     # OpenAI API call
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": (
#                     "You are a helpful career counselor AI. "
#                     "If the user does not mention their skills, hobbies, or interests, "
#                     "politely ask them to share these first before suggesting careers. "
#                     "Always reply in the same language the user uses (English or Roman Urdu)."
#                 )},
#             {"role": "user", "content": user_message}
#         ]
#     )

#     # AI ka reply
#     ai_reply = response.choices[0].message.content

#     # Response JSON me "message" field return karo
#     return jsonify({"message": ai_reply})

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input", "")

    client = get_openai_client()
    if not client:
        return jsonify({"error": "Failed to get API key"}), 500

    try:
        # OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful career counselor AI. "
                    "If the user does not mention their skills, hobbies, or interests, "
                    "politely ask them to share these first before suggesting careers. "
                    "Always reply in the same language the user uses (English or Roman Urdu)."
                )},
                {"role": "user", "content": user_input}
            ]
        )

        reply = response.choices[0].message.content
        return jsonify({"message": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/career_advice', methods=['POST'])
def career_advice():
    try:
        data = request.get_json()
        
        if not data or 'user_input' not in data:
            return jsonify({
                'error': 'user_input parameter is required',
                'success': False
            }), 400
        
        user_input = data['user_input']
        
        if not user_input.strip():
            return jsonify({
                'error': 'user_input cannot be empty',
                'success': False
            }), 400
        
        # Detect language
        language = ml_model.detect_language(user_input)
        
        # Generate response using ML model
        response = ml_model.generate_response(user_input, language)
        
        # Get career recommendations if it's a career advice request
        careers = []
        intent, confidence = ml_model.predict_intent(user_input)
        if intent == "career_advice":
            careers = ml_model.predict_careers(user_input)
        
        return jsonify({
            'success': True,
            'language_detected': language,
            'intent': intent,
            'confidence': float(confidence),
            'message': response,
            'career_recommendations': careers,
            'total_recommendations': len(careers),
            'ai_analysis': {
                'ml_model_used': 'Random Forest + TF-IDF',
                'processing_method': 'Intent Classification + Career Prediction',
                'model_confidence': float(confidence)
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'AI/ML Career Advice API is running',
        'model_status': 'trained' if ml_model.is_trained else 'not_trained',
        'models_loaded': {
            'intent_classifier': 'Random Forest',
            'career_classifier': 'Random Forest',
            'vectorizer': 'TF-IDF'
        }
    })

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Endpoint to retrain the model with new data"""
    try:
        print("Retraining model...")
        ml_model.train_model("dataset/career_dataset.json")
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully'
        })
    except Exception as e:
        return jsonify({
            'error': f'Error retraining model: {str(e)}',
            'success': False
        }), 500

@app.route('/predict_intent', methods=['POST'])
def predict_intent():
    """Endpoint to predict intent only"""
    try:
        data = request.get_json()
        
        if not data or 'user_input' not in data:
            return jsonify({
                'error': 'user_input parameter is required',
                'success': False
            }), 400
        
        user_input = data['user_input']
        language = ml_model.detect_language(user_input)
        intent, confidence = ml_model.predict_intent(user_input)
        
        return jsonify({
            'success': True,
            'user_input': user_input,
            'language_detected': language,
            'intent': intent,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        }), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'AI/ML Career Advice API',
        'version': '2.0',
        'description': 'Advanced AI-powered career guidance with trained ML models',
        'endpoints': {
            'POST /career_advice': 'Get AI-powered career recommendations with conversation handling',
            'POST /predict_intent': 'Predict user intent only',
            'POST /retrain': 'Retrain the ML model',
            'GET /health': 'Health check endpoint'
        },
        'ai_features': {
            'intent_classification': 'Random Forest classifier for conversation intents',
            'career_prediction': 'Random Forest classifier for career recommendations',
            'language_detection': 'English/Roman Urdu detection',
            'conversation_flow': 'Natural conversation handling',
            'model_persistence': 'Trained models saved and loaded automatically'
        },
        'conversation_intents': [
            'greeting', 'career_advice', 'capabilities', 'help', 'thanks', 'goodbye'
        ],
        'usage': {
            'method': 'POST',
            'url': '/career_advice',
            'body': {
                'user_input': 'Your message (English or Roman Urdu)'
            }
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
