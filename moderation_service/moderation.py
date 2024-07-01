import json
from flask import Flask, request, jsonify
from classification_model import classify_content
from sentiment_analysis import analyze_sentiment
from toxic_comment_detection import detect_toxicity

app = Flask(__name__)

@app.route('/moderate', methods=['POST'])
def moderate_content():
    data = request.get_json()
    content = data['content']
    
    classification = classify_content(content)
    sentiment = analyze_sentiment(content)
    toxicity = detect_toxicity(content)
    
    result = {
        'classification': classification,
        'sentiment': sentiment,
        'toxicity': toxicity
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
