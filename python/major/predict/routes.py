from flask_cors import cross_origin
import joblib
from flask.blueprints import Blueprint
from flask import Response, jsonify, request
import warnings
import numpy as np
predict = Blueprint('predict',__name__)

@predict.route('/opt',methods=['POST'])
def optimal_result():
    warnings.filterwarnings("ignore")
    print(request.get_json().get("model"))
    model = request.get_json().get("model")
    # for local testing
    # loaded_model = joblib.load(f'/home/kalyan/Documents/GitHub/major/python/major/predict/{model}.sav')

    # line for docker container
    loaded_model = joblib.load(f'/major/predict/{model}.sav')
    result = loaded_model.predict([request.get_json().get("values")])
    result = np.array(result).tolist()
    print(result)
    return jsonify(result)


# @predict.route('/predict',methods=['POST'])
# def predict_result():
#     warnings.filterwarnings("ignore")
#     print(request.get_json().get("model"))
#     # loaded_model = joblib.load('/home/kalyan/Documents/GitHub/major/python/major/predict/GradientBoostingClassifierModelHyperTuned.sav')
#      loaded_model = joblib.load('/major/predict/GradientBoostingClassifierModelHyperTuned.sav')
#     result = loaded_model.predict([request.get_json().get("values")])
#     result = np.array(result).tolist()
#     return result