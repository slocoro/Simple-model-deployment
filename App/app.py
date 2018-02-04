#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:11:05 2018

@author: Steven
"""

from flask import Flask, jsonify, request
import pickle
import numpy as np

# load previously trained model
filename = '../Model/Iris_model_v1.pkl'
iris_model = pickle.load(open(filename, 'rb'))

# create instance of flask app
app = Flask(__name__)
app.debug = True


@app.route('/predict', methods=['POST'])
def predict():
    
    # get data from post request
    data = request.get_json(force=True)
    
    # parse data for individual features, cast to numpy array and reshape
    predict_request = [data['sl'], data['sw'], data['pl'], data['pw']]
    predict_request = np.array(predict_request).reshape(1, -1)
    
    # perform prediction using previously trained model
    y_pred = iris_model.predict(predict_request)
    
    # cast prediction to string to be able to pass it to jsonify
    output = str([y_pred[0]])
    
    return jsonify(results=output)


# if this condition met this script will be executed
if __name__ == '__main__':

	app.secret_key = 'secret123'
	app.run()
    
    
    
    
    
    
    
    
    
    