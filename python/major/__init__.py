from flask import Flask



def create_app():
    app=Flask(__name__)
    
    from major.predict.routes import predict

    app.register_blueprint(predict)
    
    return app