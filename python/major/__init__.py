from flask import Flask
from flask_cors import CORS


def create_app():
    app=Flask(__name__)
    CORS(app)
    
    from major.predict.routes import predict

    app.register_blueprint(predict)
    
    return app