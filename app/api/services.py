from flask import Blueprint, Response, request, json
import time
from ..database.dbManager import DBManager
from ..database.authManager import AuthManager
from ..portfolio.portfolioOverview import PortfolioOverview
from ..utilities.generalUtils import GeneralUtils
import requests
from .. import socketio
from ..streamer.newsStreamer import NewsStreamer
import configparser
import pandas as pd


config = configparser.ConfigParser()
config.read('config.ini')

apsendysAPI = Blueprint('apsendysAPI', __name__)
endpoint = "/api"

db_manager = DBManager()
auth_manager = AuthManager()
porfolio_overview = PortfolioOverview()
utils = GeneralUtils()
news = NewsStreamer()

@apsendysAPI.route("/error", methods=['POST'])
def error():
    try:
        data = json.loads(request.data.decode())
        response = Response(json.dumps({"error": data}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getReportsSentiment/email/<string:email>", methods=['GET'])
def getInvestmentsReportsSentiment(email):
    result = []
    try:
        user = auth_manager.retrieve_user_by_email(email)
        investment = db_manager.get_reports_sentiment(user["user_id"], "investment")
        watchlist = db_manager.get_reports_sentiment(user["user_id"], "watchlist")
        response = Response(json.dumps({"investment": investment, "watchlist": watchlist}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getAllAssets", methods=['GET'])
def getAllAssets():
    result = {}
    try:
        assets = db_manager.get_all_assets()
        response = Response(json.dumps({"assets": assets}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getOptAndProfileOptions", methods=['GET'])
def getOptAndProfileOptions():
    result = {}
    try:
        opt_methods = db_manager.get_all_opt_methods()
        risk_profiles = db_manager.get_all_risk_profiles()
        response = Response(json.dumps({"opt_methods": opt_methods, "risk_profiles": risk_profiles}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getAssetsSentiment/email/<string:email>", methods=['GET'])
def getAssetsSentiment(email):
    result = []
    try:
        user = auth_manager.retrieve_user_by_email(email)        
        result = db_manager.get_user_assets_sentiment(user["user_id"])
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getSentimentChange/email/<string:email>", methods=['GET'])
def getInvestmentsSentimentChange(email):
    result = []
    try:
        user = auth_manager.retrieve_user_by_email(email)
        investment = db_manager.get_sentiment_change(user["user_id"], "investment")
        watchlist = db_manager.get_sentiment_change(user["user_id"], "watchlist")
        response = Response(json.dumps({"investment": investment, "watchlist": watchlist}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getPortfolioStatistics/email/<string:email>", methods=['GET'])
def getPortfolioStatistics(email):
    result = {}        
    try:

        user = auth_manager.retrieve_user_by_email(email)
        response = db_manager.get_user_portfolio_and_assets(user["user_id"])

        portfolio_register_date = db_manager.get_portfolio_register_date(user["user_id"])
        portfolioValue_resp = porfolio_overview.calcPortfolioValueByDate(user["user_id"], portfolio_register_date) 
        resp = {}

        curr_val = porfolio_overview.calcPortfolioCurrentValue(user["user_id"])
        resp["inital_value"] = portfolioValue_resp["portfolioValue"] if portfolioValue_resp["status"] == "SUCCESS" else None
        resp["current_value"] = curr_val["portfolioCurrValue"] if curr_val["status"] == "SUCCESS" else None
        resp["total"] = porfolio_overview.calcPortfolioProfit(resp["inital_value"], resp["current_value"]) if resp["current_value"] != None else None
        resp["total_percentage"] = porfolio_overview.calcPortfolioProfitPercentage(resp["inital_value"], resp["current_value"]) if resp["current_value"] != None else None

        assets = response['assets']
        watchlist = response['watchlist']
        total_value = response['assets_value'] + response['cash']        
        
        result["assets_value"] = response['assets_value']
        result["cash"] = response['cash']
        result["total_value"] = response['assets_value'] + response['cash']
        result["investment_percentage"] = response['investment_percentage']
        result["cash_percentage"] = 100 - response['investment_percentage']
        result["total_percentage"] = resp["total_percentage"]

        assets_df = pd.DataFrame(assets)
        assets_df["total_pieces_value"] = assets_df["current_value"]*assets_df["asset_pieces_number"]
        sentiment_groups = assets_df.groupby(["current_sentiment"]).sum().reset_index()

        strong_positive = {}
        positive = {}
        neutral = {}
        negative = {}
        strong_negative = {}     
        cash = {'sentiment': 'cash', 'value': response['cash'], 'percentage': result["cash_percentage"], 'color': '#D3D3D3' }

        if len(sentiment_groups[sentiment_groups["current_sentiment"]==2.0]) > 0:
            data = sentiment_groups[sentiment_groups["current_sentiment"]==2.0]
            strong_positive = { 'sentiment': 'very_positive', 'value': float(data["total_pieces_value"]), 'percentage': float(data["percentage"]), 'color': '#90EE90' }          
        else: 
            strong_positive = { 'sentiment': 'very_positive', 'value': 0, 'percentage': 0, 'color': '#90EE90' }
        
        if len(sentiment_groups[sentiment_groups["current_sentiment"]==1.0]) > 0:
            data = sentiment_groups[sentiment_groups["current_sentiment"]==1.0]
            positive = { 'sentiment': 'positive', 'value': float(data["total_pieces_value"]), 'percentage': float(data["percentage"]), 'color': '#008000' }           
        else: 
            positive = { 'sentiment': 'positive', 'value': 0, 'percentage': 0, 'color': '#008000' }

        if len(sentiment_groups[sentiment_groups["current_sentiment"]==0.0]) > 0:
            data = sentiment_groups[sentiment_groups["current_sentiment"]==0.0]
            neutral = { 'sentiment': 'neutral', 'value': float(data["total_pieces_value"]), 'percentage': float(data["percentage"]), 'color': '#FF8C00' }       
        else: 
            neutral = { 'sentiment': 'neutral', 'value': 0, 'percentage': 0, 'color': '#FF8C00' }

        if len(sentiment_groups[sentiment_groups["current_sentiment"]==-1.0]) > 0:
            data = sentiment_groups[sentiment_groups["current_sentiment"]==-1.0]
            negative = { 'sentiment': 'negative', 'value': float(data["total_pieces_value"]), 'percentage': float(data["percentage"]), 'color': '#FFCCCB' }           
        else: 
            negative = { 'sentiment': 'negative', 'value': 0, 'percentage': 0, 'color': '#FFCCCB' }

        if len(sentiment_groups[sentiment_groups["current_sentiment"]==-2.0]) > 0:
            data = sentiment_groups[sentiment_groups["current_sentiment"]==-2.0]
            strong_negative = { 'sentiment': 'very_negative', 'value': float(data["total_pieces_value"]), 'percentage': float(data["percentage"]), 'color': '#FF0000' }         
        else: 
            strong_negative = { 'sentiment': 'very_negative', 'value': 0, 'percentage': 0, 'color': '#FF0000' }

        all_stats = [strong_positive, positive, neutral, negative, strong_negative, cash]

        result["all_stats"] = all_stats
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getActiveSignals/email/<string:email>", methods=['GET'])
def getSignals(email):
    result = []
    try:
        user = auth_manager.retrieve_user_by_email(email)
        result = db_manager.get_active_signals_by_user_id(user["user_id"])
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getUserPortfolio/email/<string:email>", methods=['GET'])
def getUserPortfolio(email):
    result = []
    try:
        user = auth_manager.retrieve_user_by_email(email)
        result = db_manager.get_user_portfolio_and_assets(user["user_id"])
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getAssetIndices/assetId/<string:assetId>", methods=['GET'])
def getAssetIndices(assetId):    
    result = []    
    try:               
        data = json.loads(request.data.decode())
        result = db_manager.get_asset_indices(assetId, data["from"], data["to"])
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getAcceptedSignalDetails/accepted_signal_id/<string:accepted_signal_id>", methods=['GET'])
def getAcceptedSignalDetails(accepted_signal_id):    
    result = {}
    try:
        details = db_manager.get_signals_acceptance_history_by_id(int(accepted_signal_id))   
        if details.empty == False:
            details = details.iloc[0]
            details = details.fillna(int(-1))
            result = {
                "asset_profit": details["asset_profit_percentage"],
                "portfolio_profit": details["portfolio_asset_profit_percentage"],
                "buy_value": details["asset_buy_value"],
                "sell_value": details["asset_sell_value"],
                "current_value": db_manager.getAssetCurrValue(details["asset_id"]),
                "current_investment_sentiment": db_manager.getAssetCurrSentiment(details["asset_id"])["sentiment_text"]
            }        
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getAcceptedInvestmentSignals/email/<string:email>", methods=['GET'])
def getAcceptedInvestmentSignals(email):    
    result = []   

    try:
        user = auth_manager.retrieve_user_by_email(email)
        signals_history = db_manager.get_signals_acceptance_history(user["user_id"])
        signals_history = signals_history.fillna(-1)
        result = signals_history.to_dict('records')
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getPortfolioOverview/email/<string:email>/from_date/<string:from_date>", methods=['GET'])
def getPortfolioOverview(email, from_date):    
    result = []   
    #BUG kati paei lathos sthn vdomada 
    try:         
        user = auth_manager.retrieve_user_by_email(email)   
        from_date = int(from_date)
        to_date = utils.getCurrentTimestsamp()
        portfolio_register_date = db_manager.get_portfolio_register_date(user["user_id"])
        if from_date < portfolio_register_date:
            from_date = portfolio_register_date
        result = porfolio_overview.getPortfolioOverview(user["user_id"], from_date, to_date)
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getPortfolioHistoricalTotal/email/<string:email>/from_date/<string:from_date>", methods=['GET'])
def getPortfolioHistoricalTotal(email, from_date):    
    result = []    
    try:
        user = auth_manager.retrieve_user_by_email(email)
        from_date = int(from_date)
        to_date = utils.getCurrentTimestsamp()
        portfolio_register_date = db_manager.get_portfolio_register_date(user["user_id"])
        if from_date < portfolio_register_date:
            from_date = portfolio_register_date
        result = porfolio_overview.getHistoricalPortfolioProfit(user["user_id"], from_date, to_date)
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getAssetHistory/asset_id/<string:asset_id>", methods=['GET'])
def getAssetHistory(asset_id):    
    result = []    
    dist = 3
    try:              
        result = db_manager.get_indeces_history_by_asset(asset_id, 100)
        result["asset"] = db_manager.get_asset_info(asset_id)
        signals = db_manager.get_signals_by_asset(asset_id, 100)
        result["techical_buy"] = {'value': [None] * len(result['label']), 'style': [None] * len(result['label'])}
        result["techical_sell"] = {'value': [None] * len(result['label']), 'style': [None] * len(result['label'])}
        result["sentiment_buy"] = {'value': [None] * len(result['label']), 'style': [None] * len(result['label'])}
        result["sentiment_sell"] = {'value': [None] * len(result['label']), 'style': [None] * len(result['label'])}
        result["ml_buy"] = {'value': [None] * len(result['label']), 'style': [None] * len(result['label'])}
        result["ml_sell"] = {'value': [None] * len(result['label']), 'style': [None] * len(result['label'])}
        result["mixed_buy"] = {'value': [None] * len(result['label']), 'style': [None] * len(result['label'])}
        result["mixed_sell"] = {'value': [None] * len(result['label']), 'style': [None] * len(result['label'])}
        for i in range(len(result['label'])):
            for j in range(len(signals['date'])):
                if signals["type"][j] == 1 and signals["action"][j] == 1 and signals["date"][j] == result['label'][i]: #technical - buy
                    result["techical_buy"]["value"][i] = result['value'][i] + (result['value'][i]*dist/100)
                    result["techical_buy"]["style"][i] = "rectRot"
                    # break
                elif  signals["type"][j] == 1 and signals["action"][j] == 2 and signals["date"][j] == result['label'][i]:  #technical - sell
                    result["techical_sell"]["value"][i] = result['value'][i] - (result['value'][i]*dist/100)
                    result["techical_sell"]["style"][i] = "rectRot"
                    # break
                elif  signals["type"][j] == 2 and signals["action"][j] == 1 and signals["date"][j] == result['label'][i]:  #sentiment - buy
                    result["sentiment_buy"]["value"][i] = result['value'][i] + (result['value'][i]*dist/100)
                    result["sentiment_buy"]["style"][i] = "circle"
                    # break
                elif  signals["type"][j] == 2 and signals["action"][j] == 2 and signals["date"][j] == result['label'][i]:  #sentiment - sell
                    result["sentiment_sell"]["value"][i] = result['value'][i] - (result['value'][i]*dist/100)
                    result["sentiment_sell"]["style"][i] = "circle"
                    # break
                elif  signals["type"][j] == 3 and signals["action"][j] == 1 and signals["date"][j] == result['label'][i]:  #ml - buy
                    result["ml_buy"]["value"][i] = result['value'][i] + (result['value'][i]*dist/100)
                    result["ml_buy"]["style"][i] = "rect"
                    # break
                elif  signals["type"][j] == 3 and signals["action"][j] == 2 and signals["date"][j] == result['label'][i]:  #ml - sell
                    result["ml_sell"]["value"][i] = result['value'][i] - (result['value'][i]*dist/100)
                    result["ml_sell"]["style"][i] = "rect"
                    # break
                elif  signals["type"][j] == 4 and signals["action"][j] == 1 and signals["date"][j] == result['label'][i]:  #mixed - buy
                    result["mixed_buy"]["value"][i] = result['value'][i] + (result['value'][i]*dist/100)
                    result["mixed_buy"]["style"][i] = "triangle"
                    # break
                elif  signals["type"][j] == 4 and signals["action"][j] == 2 and signals["date"][j] == result['label'][i]:  #mixed - sell
                    result["mixed_sell"]["value"][i] = result['value'][i] - (result['value'][i]*dist/100)
                    result["mixed_sell"]["style"][i] = "triangle"
                    # break
        # result = {
        #     'label': ['01/19', '011/19', '02/19', '03/19', '05/19', '05/19', '06/19', '07/19', '08/19', '09/19', '10/19', '11/19', '12/19', '01/20', '02/20', '03/20', '04/20', '05/20', '06/20', '07/20', '08/20'],
        #     'value': [45000, 59000, 55000, 80000, 81000, 56000, 55000, 95000, 64000, 32000, 79000, 82000, 43000, 56000, 55000, 95000, 64000, 32000, 79000, 82000, 43000],
        #     'ml_buy': {
        #         'value': [None, None, 60000, None, None, None, 60000, None, None, None, 84000, None, 48000, None, 60000, None, None, None, None, 87000, None],
        #         'style': [None, None, "circle", None, None, None, "circle", None, None, None, "circle", None, "circle", None, "circle", None, None, None, None, "rectRot", None]
        #     },
        #     'ml_sell': {
        #         'value': [40000, None, None, None, None, None, None, 90000, None, None, None, 77000, None, None, None, None, 79000, None, None, None, None],
        #         'style': ["circle", None, None, None, None, None, None, "circle", None, None, None, "circle", None, None, None, None, "rectRot", None, None, None, None]
        #     }
        # }
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/optimizePortfolio", methods=['POST'])
def optimizePortfolio():
    try:
        data = json.loads(request.data.decode())
        
        r = requests.post(config['optimizer']['protocol'] + "://" + config['optimizer']['ip'] + ":" +  config['optimizer']['port'] + "/port_opt", json=data)
        result = json.dumps(r.json())
        response = Response(result, mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getAllAssetsWithCurrentValues", methods=['GET'])
def getAllAssetsWithCurrentValues():    
    result = []    
    try:               
        result = porfolio_overview.getAllAssetsWithCurrentValues()
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/sellPortfolioAsset/email/<string:email>/asset_id/<string:asset_id>", methods=['GET'])
def sellPortfolioAsset(email, asset_id):    
    result = []    
    try:       
        user = auth_manager.retrieve_user_by_email(email)        
        result = porfolio_overview.sellAsset(user["user_id"], int(asset_id))
        socketio.emit('update_portfolio', {"update": True})
        response = Response(json.dumps(result), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response  

@apsendysAPI.route(endpoint + "/buyPortfolioAsset/email/<string:email>/asset_id/<string:asset_id>/budget/<string:budget>", methods=['GET'])
def buyPortfolioAsset(email, asset_id, budget):    
    result = []    
    try:               
        user = auth_manager.retrieve_user_by_email(email)
        result = porfolio_overview.buyAsset(user["user_id"], int(asset_id), float(budget))
        response = Response(json.dumps({"data": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response  

@apsendysAPI.route(endpoint + "/addWatchlistAsset/email/<string:email>/asset_id/<string:asset_id>", methods=['GET'])
def addWatchlistAsset(email, asset_id):    
    result = []    
    try:               
        user = auth_manager.retrieve_user_by_email(email)
        result = db_manager.add_watchlist_asset(user["user_id"], asset_id)
        response = Response(json.dumps({"success": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response 

@apsendysAPI.route(endpoint + "/deleteWatchlistAsset/email/<string:email>/asset_id/<string:asset_id>", methods=['GET'])
def deleteWatchlistAsset(email, asset_id):    
    result = []    
    try:               
        user = auth_manager.retrieve_user_by_email(email)
        result = db_manager.delete_watchlist_asset(str(user["user_id"]), asset_id)
        response = Response(json.dumps({"success": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response 

@apsendysAPI.route(endpoint + "/saveAcceptedSignals/email/<string:email>", methods=['POST'])
def saveAcceptedSignals(email):
    try:
        all_signals = json.loads(request.data.decode())
        user = auth_manager.retrieve_user_by_email(email)
        for signal in all_signals:
            if int(signal["signal_action_id"]) == 1:
                porfolio_overview.buyAsset(user["user_id"], signal["asset_id"], signal["budget"], signal["signal_id"])
            elif int(signal["signal_action_id"]) == 2:
                porfolio_overview.sellAsset(user["user_id"], signal["asset_id"])
        
        socketio.emit('update_portfolio', {"update": True})
        socketio.emit('update_signals', {"update": True})
        response = Response(json.dumps({"success": True}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/getAssetCurrentValue/asset_id/<string:asset_id>", methods=['GET'])
def getAssetCurrentValue(asset_id):    
    result = []    
    try:               
        value = db_manager.getAssetCurrValue(int(asset_id))
        response = Response(json.dumps({"success": True, "value": value}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

@apsendysAPI.route(endpoint + "/postSignals", methods=['POST'])
def postSignals():
    result = {}
    try:
        data = json.loads(request.data.decode())
        result = db_manager.insert_signals(data)
        socketio.emit('new_signals', result)
        response = Response(json.dumps({"result": result}), mimetype='application/json', status=200)
    except Exception as e:
        response = handleException(e)
    finally:
        return response

def handleException(e):
    result = json.dumps({"error": str(e)})
    print("=======================================" + result + "=======================================" )
    return Response(result, mimetype='application/json', status=500)