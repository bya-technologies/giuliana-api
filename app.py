from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Baseten API configuration
BASETEN_API_KEY = os.environ.get('BASETEN_API_KEY')
BASETEN_MODEL_ID = "q6on32g"  # DeepSeek-V3 model ID
BASETEN_API_URL = f"https://model-{BASETEN_MODEL_ID}.api.baseten.co/production/predict"

@app.route('/api/giuliana', methods=['POST'])
def giuliana_chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('max_tokens', 2000)
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        # Call Baseten DeepSeek API
        headers = {
            'Authorization': f'Api-Key {BASETEN_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False
        }
        
        response = requests.post(BASETEN_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        assistant_message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        return jsonify({'response': assistant_message})
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'API request failed: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
