from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize the text generation pipeline with GPT-Neo
pipe = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data.get('input', '')

    # Generate a response from the model
    response = pipe(input_text, max_length=50, num_return_sequences=1)

    # Return the generated text as JSON
    return jsonify({'output': response[0]['generated_text']})

if __name__ == '__main__':
    # Run the Flask app on the specified host and port
    app.run(host='0.0.0.0', port=5000)
