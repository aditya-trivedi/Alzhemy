from flask import Flask, render_template
from flask import request
import flask
from pycaret.classification import *
import numpy as np
import pandas as pd

app = flask.Flask(__name__)
model = load_model('lr_model')
cols = ['AGE','PTGENDER','PTEDUCAT','PTETHCAT' ,'PTRACCAT' , 'APOE4','MMSE','imputed_genotype','APOE Genotype']
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    input_values = [x for x  in request.form.values()]
    final = np.array(input_values)
    data_unseen = pd.DataFrame([final],columns = cols)
    prediction  = predict_model(model , data = data_unseen )
    print(prediction.Label[0])
    return  prediction.Label[0]


app.run()
