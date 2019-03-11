#from typing import Iterator, List, Dict
import numpy as np
from TorchMoviePredictor import *

from flask import Flask, render_template, request
app = Flask(__name__)


user_name = ""

print('Predictor is loading...')
predictor = get_predictor()
print('Predictor Loaded!')

@app.route('/')
def sentence_input():
    return render_template('input_screen.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        global user_name
        result = request.form
        if 'Name' in result:
            user_name = result['Name']
        text_review = result['ReviewText']
        logits = predictor.predict(preprocess_text(text_review))['logits']
        prediction = round(max(logits)*100,2)
        print(prediction)
        label_id = np.argmax(logits)
        #prediction = model.vocab.get_token_from_index(label_id, 'labels')
        print(label_id,'  ',prediction)
        if label_id == 1:
            prediction_text = "It looks like you liked this movie, I feel " + str(prediction) + "% sure."
        else:
            prediction_text = "It does not seem like you liked this movie, I feel " + str(prediction) + "% sure."
        
        render_dict = {"Name":user_name,'Review Text':result['ReviewText'],"Sentiment Prediction":prediction_text}
        return render_template("result.html",result = render_dict)

if __name__ == '__main__':
    app.run(debug = True)