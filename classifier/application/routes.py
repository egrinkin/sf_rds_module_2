from flask import Flask, request, jsonify
from application import app
from spam_classifier import Classifier

@app.route('/classify_text', methods=['POST'])

def classify_text():
    data = request.json
    text = data.get('text')
    sp_cl = Classifier()
    result = sp_cl.classify(text)
    if text is None:
        params = ', '.join(data.keys()) 
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400 
    else:
        return jsonify({'result': result})


    
        
