from flask import Blueprint, Response, request, json, current_app
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity, decode_token

from ..database.authManager import AuthManager
from .. import socketio
from datetime import timedelta

authAPI = Blueprint('authAPI', __name__)
endpoint = "/auth"
access_token_expiration_time = 3600 #seconds
refresh_token_expiration_time = 15 #days

auth_manager = AuthManager()



@authAPI.route(endpoint + "/login", methods=['POST'])
def login():
    result = {}
    try:
        
        data = json.loads(request.data.decode())
        user = auth_manager.retrieve_user(data)
        if len(user) > 0:
            result = create_jwt_payload(user)
            response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
        else:
            response = Response(json.dumps({"data": []}), mimetype='application/json', status=401)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@authAPI.route(endpoint + "/refresh-token", methods=['POST'])
def refresh_token():
    try:
        previous_refresh_token = decode_token(json.loads(request.data.decode())["token"]["refresh_token"])
        result = create_jwt_payload(previous_refresh_token["identity"])
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@authAPI.route(endpoint + "/logout", methods=['DELETE'])
@jwt_required
def logout():
    try:
        result = json.dumps({})
        response = Response(result, mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@authAPI.route(endpoint + "/register", methods=['POST'])
def register():
    try:
        data = json.loads(request.data.decode())
        user = auth_manager.store_user(data)
        result = create_jwt_payload(user)
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@authAPI.route(endpoint + "/forgot_password", methods=['POST'])
def forgot_password():
    try:
        data = json.loads(request.data.decode())
        result = json.dumps(data)
        response = Response(result, mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response


def create_jwt_payload(data):
    access_token = create_access_token(identity=data, expires_delta=timedelta(seconds=access_token_expiration_time)) 
    refresh_token = create_refresh_token(identity=data, expires_delta=timedelta(days=refresh_token_expiration_time))

    payload = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": access_token_expiration_time,
            "token_type": "Bearer",
            }
    return payload

@authAPI.route("/api/updatePortfolio", methods=['POST'])
def updatePortfolio():
    try:
        data = json.loads(request.data.decode())
        user = auth_manager.retrieve_user_by_email(data["user_email"])
        updated = auth_manager.edit_user_portfolio(data, user["user_id"])
        socketio.emit('update_portfolio', {"update": True})
        response = Response(json.dumps({"success": updated}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@authAPI.route("/api/saveInitialPortfolio", methods=['POST'])
def saveInitialPortfolio():
    try:
        data = json.loads(request.data.decode())
        auth_manager.save_user_portfolio(data)
        result = json.dumps({"saved": True})
        response = Response(result, mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

def handleException(e):
    result = json.dumps({"error": str(e)})
    print("=======================================" + result + "=======================================" )
    return Response(result, mimetype='application/json', status=500)
