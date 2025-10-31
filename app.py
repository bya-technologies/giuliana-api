from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "Giuliana AI Concierge",
        "model": "deepseek-chat",
        "version": "1.0"
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
        
        # Call DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return jsonify({
            "response": response.choices[0].message.content,
            "model": "deepseek-chat",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
