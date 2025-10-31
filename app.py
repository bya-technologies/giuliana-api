from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "Giuliana AI Concierge",
        "model": "deepseek-chat",
        "version": "1.1"
    })

@app.route('/api/giuliana', methods=['POST'])
def giuliana():
    try:
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({"error": "Messages are required"}), 400
        
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('max_tokens', 2000)
        
        # Call DeepSeek API directly using requests
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        return jsonify({
            "response": result['choices'][0]['message']['content'],
            "model": "deepseek-chat",
            "usage": result.get('usage', {})
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
