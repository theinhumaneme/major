import joblib
from flask.blueprints import Blueprint
from flask import request
import numpy as np
import json
predict = Blueprint('predict',__name__)

import warnings
warnings.filterwarnings(category=UserWarning)

@predict.route('/opt',methods=['POST'])
def optimal_result():
    loaded_model = joblib.load("/major/models/GradientBoostingClassifierModelHyperTuned.sav")
    result = loaded_model.predict([[request.data.get("values")]])
    np.array(result)
    result = result.tolist()
    return json.dumps(result)