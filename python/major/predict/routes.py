import joblib
from flask.blueprints import Blueprint
from flask import request
import warnings
import numpy as np
predict = Blueprint('predict',__name__)

@predict.route('/opt',methods=['GET','POST'])
def optimal_result():
    if request.method == "GET":
        return "hello world"
    # print(request.data)
    warnings.filterwarnings("ignore")
    # print(request.get_json().get("values"))
    loaded_model = joblib.load('/home/kalyan/Documents/GitHub/major/python/major/predict/GradientBoostingClassifierModelHyperTuned.sav')
    result = loaded_model.predict([request.get_json().get("values")])
    result = np.array(result).tolist()
    return result


@predict.route('/predict',methods=['POST'])
def predict_result():
    if request.method == "GET":
        return "hello world"
    # print(request.data)
    warnings.filterwarnings("ignore")
    # print(request.get_json().get("values"))
    loaded_model = joblib.load('/home/kalyan/Documents/GitHub/major/python/major/predict/GradientBoostingClassifierModelHyperTuned.sav')
    result = loaded_model.predict([request.get_json().get("values")])
    result = np.array(result).tolist()
    return result