from flask import Flask, render_template, request, jsonify
from word_count import count_words
from flask_cors import CORS
from fakeTweetPredictor import returnPrediction

##########################################

from transformers import logging
logging.set_verbosity_error()

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/count-words', methods=['POST'])
def get_word_count():
    data = request.json
    text = data['text']
    word_count = count_words(text)
    word_count = returnPrediction(text)
    return jsonify({'word_count': word_count})

if __name__ == '__main__':
    app.run(debug=True)
