import mysql.connector as mySQL
import configparser
import hashlib
import time
from datetime import datetime, timedelta
import calendar
import pandas as pd
import json


from ..utilities.generalUtils import GeneralUtils

config = configparser.ConfigParser()
config.read('config.ini')

general_utils = GeneralUtils()

class DBManager:

    def getDatabaseConnection(self):
        return mySQL.connect(host=config['database']['host'], user=config['database']['user'], passwd=config['database']['passwd'], db=config['database']['db'], charset=config['database']['charset'], auth_plugin='mysql_native_password')


    def get_all_assets(self):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT * FROM assets"
            cursor.execute(select_query)
            all_assets = cursor.fetchall()
            for asset in all_assets:
                result.append({"asset_id": asset[0], "name": asset[1], "ticker": asset[2],"yahoo_ticker": asset[5]})
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result

    def get_asset_tickers(self):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            cursor.execute("SELECT ticker,asset_id,name FROM assets")
            results = cursor.fetchall()
            resp = [(item[0], item[1], item[2]) for item in results]
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = []
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_asset_twitter_queries(self, asset_id):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            cursor.execute("SELECT twitter_query FROM assets WHERE asset_id = " + str(asset_id))
            results = cursor.fetchall()
            query_list = results[0][0].split(",")
            resp = [q.strip() for q in query_list]
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = []
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_asset_stocktwits_query(self, asset_id):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            cursor.execute("SELECT stocktwits_query FROM assets WHERE asset_id = " + str(asset_id))
            results = cursor.fetchall()
            resp = results[0][0]
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = []
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_all_opt_methods(self):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT * FROM opt_methods"
            cursor.execute(select_query)
            all_methods = cursor.fetchall()
            for method in all_methods:
                result.append({"opt_method_id": method[0], "value": method[1], "name": method[2]})
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result

    def get_all_risk_profiles(self):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT * FROM risk_profiles"
            cursor.execute(select_query)
            all_profiles = cursor.fetchall()
            for profile in all_profiles:
                result.append({"risk_profile_id": profile[0], "name": profile[1], "value": profile[2], "status": profile[3], "description": profile[4]})
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result

    def get_user_portfolio_and_assets(self, user_id):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()

            select_query =  " SELECT portfolio.portfolio_id, portfolio.cash, portfolio.portfolio_assets_value, portfolio.investment_percentage, opt_methods.*, risk_profiles.*"
            select_query += " FROM portfolio"
            select_query += " INNER JOIN opt_methods"
            select_query += " ON portfolio.optimization_method = opt_methods.opt_method_id"
            select_query += " INNER JOIN risk_profiles"
            select_query += " ON portfolio.profile_type = risk_profiles.risk_profile_id"
            select_query += " WHERE portfolio.user_id = " + str(user_id)
            cursor.execute(select_query)
            portfolio = cursor.fetchall()
            if len(portfolio) > 0:
                response = {
                    "portfolio_id": portfolio[0][0],   
                    "cash": portfolio[0][1],
                    "assets_value": portfolio[0][2],
                    "investment_percentage": portfolio[0][3],
                    "optimization_method":{
                        "opt_method_id": portfolio[0][4],
                        "value": portfolio[0][5],
                        "name": portfolio[0][6],
                    },
                    "risk_profile": {
                        "risk_profile_id": portfolio[0][7],
                        "name": portfolio[0][8],
                        "value": portfolio[0][9],
                    }  
                }
            
            select_query2 = " SELECT assets.asset_id, assets.name, assets.ticker, assets.description, assets.class,  assets.yahoo_ticker, SUM(user_assets.percentage), SUM(user_assets.amount) "
            select_query2 +=" FROM user_assets"
            select_query2 +=" INNER JOIN assets"
            select_query2 +=" ON assets.asset_id = user_assets.asset_id AND user_assets.user_id = " + str(user_id)
            select_query2 +=" GROUP BY user_assets.asset_id"

            cursor.execute(select_query2)
            all_assets = cursor.fetchall()
            assets = []
            if len(all_assets) > 0:
                for asset in all_assets:
                    asset_id = asset[0]
                    current_value = self.getAssetCurrValue(asset_id)
                    current_sentiment = self.getAssetCurrSentiment(asset_id)["sentiment"]
                    assets.append({
                        "asset_id": asset_id,
                        "name": asset[1],
                        "ticker": asset[2],
                        "description": asset[3],
                        "class": asset[4],
                        "yahoo_ticker": asset[5],
                        "percentage": asset[6],
                        "current_value": current_value,
                        "current_sentiment": current_sentiment,
                        "asset_pieces_number": asset[7]
                    })
            response["assets"] = assets

            select_query3 =  " SELECT assets.asset_id, assets.name, assets.ticker, assets.description, assets.class, assets.yahoo_ticker "
            select_query3 += " FROM assets "
            select_query3 += " INNER JOIN user_watchlist"
            select_query3 +=  " ON user_watchlist.asset_id = assets.asset_id AND user_watchlist.user_id = " + str(user_id)
            cursor.execute(select_query3)
            watchlist_assets = cursor.fetchall()
            watchlist = []
            if len(watchlist_assets) > 0:
                for asset in watchlist_assets:
                    asset_id = asset[0]
                    current_value = self.getAssetCurrValue(asset_id)
                    current_sentiment = self.getAssetCurrSentiment(asset_id)["sentiment"]
                    watchlist.append({
                        "asset_id": asset_id,
                        "name": asset[1],
                        "ticker": asset[2],
                        "description": asset[3],
                        "class": asset[4],
                        "current_value": current_value,
                        "current_sentiment": current_sentiment,
                        "yahoo_ticker": asset[5],
                    })
            response["watchlist"] = watchlist
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return response

    def getAssetCurrValue(self, assetId):
        current_milli_time = general_utils.getCurrentTimestsamp()
        from_msecs = current_milli_time - general_utils.convertDaysToMiliseconds(1000)    
        asset_curr_value = self.get_asset_indices(assetId, from_msecs, current_milli_time) 
        if "asset_indices" in asset_curr_value and len(asset_curr_value["asset_indices"]) > 0:
            #pairnw shmerinh kai 30-meres prin timh se periptwsh pou den exw shmerinh na douleve me palia
            # an den exei kanena apo ta 2 lew oti den einai updated oi times tis metoxhs
            df = pd.DataFrame(asset_curr_value["asset_indices"])
            close = float(df.nlargest(1, ['date'])["close"])
            return close
        else:
            return None
    
    def getAssetCurrSentiment(self, assetId):
        current_milli_time = general_utils.getCurrentTimestsamp()
        from_msecs = current_milli_time - general_utils.convertDaysToMiliseconds(1000)    
        asset_curr_sent = self.get_asset_sentiment(assetId, from_msecs, current_milli_time) 
        if "name" in asset_curr_sent:
            return asset_curr_sent
        else:
            return None

    def get_user_history(self, user_id, from_date, to_date):
        try:
            from_date_start, from_date_end = general_utils.getStartEndFromDate(from_date)
            to_date_start, to_date_end = general_utils.getStartEndFromDate(to_date)
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT * "
            select_query += " FROM history"
            select_query += " WHERE history.user_id = " + str(user_id)
            select_query += " AND date >= " + str(from_date_start)
            select_query += " AND date <= " + str(to_date_end)
            # select_query += " AND asset_id IS NOT NULL"
            
            cursor.execute(select_query)
            response = cursor.fetchall()
            user_history = general_utils.sql_to_dataframe(response, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
            user_history = None
        finally:
            cursor.close()
            mydb.close()
            return user_history

    def addAssetInPortfolio(self, user_id, asset_id, date, numOfAssets, asset_curr_val):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()            
            sql = "INSERT INTO user_assets (user_id, asset_id, percentage, date, amount, asset_buy_value)" \
                " VALUES (%s, %s, %s, %s, %s, %s)"
            values = (user_id, asset_id, None, date, numOfAssets, asset_curr_val)
            cursor.execute(sql, values)
            mydb.commit()
            resp = { "status": "SUCCESS" } 
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp
    
    def removeAssetFromPortfolio(self, user_id, asset_id):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()                                    

            delete_query =  " DELETE FROM user_assets "
            delete_query += " WHERE user_id = " + str(user_id)
            delete_query += " AND asset_id = " + str(asset_id)

            cursor.execute(delete_query)    
            mydb.commit()        
            resp = { "status": "SUCCESS" }
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp
    
    def updateUserCash(self, user_id, new_cash_value):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            update_query =  " UPDATE portfolio "
            update_query += " SET cash = " + str(new_cash_value)
            update_query += " WHERE user_id = " + str(user_id)

            cursor.execute(update_query)    
            mydb.commit()        
            resp = { "status": "SUCCESS" }
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" }
        finally:
            cursor.close()
            mydb.close()
            return resp
    
    def get_user_invested_money(self, user_id):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT portfolio.invested_money "
            select_query += " FROM portfolio"
            select_query += " WHERE portfolio.user_id = " + str(user_id)

            cursor.execute(select_query)
            user_budget = cursor.fetchall()            
            resp = { "cash": user_budget[0][0] }                                           
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_user_budget(self, user_id):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT portfolio.cash "
            select_query += " FROM portfolio"
            select_query += " WHERE portfolio.user_id = " + str(user_id)

            cursor.execute(select_query)
            user_budget = cursor.fetchall()            
            resp = { "cash": user_budget[0][0] }                                           
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return resp

    
    def get_user_budget_by_date(self, user_id, date):
        try:            
            start, end = general_utils.getStartEndFromDate(date)
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT cash "
            select_query += " FROM portfolio_history"
            select_query += " WHERE user_id = " + str(user_id)
            select_query += " AND date >= " + str(start) + " AND date <= " + str(end)
            cursor.execute(select_query)
            response = cursor.fetchall()   

            result = general_utils.sql_to_json(response, cursor)
            if len(result)>0:
                resp = result[0]                                            
        except Exception as e:            
            print("====================" + str(e) + "====================")
            resp = None
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_portfolio_history(self, user_id, from_date, to_date):
        try:
            from_date_start, from_date_end = general_utils.getStartEndFromDate(from_date)
            to_date_start, to_date_end = general_utils.getStartEndFromDate(to_date)
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT * "
            select_query += " FROM portfolio_history"
            select_query += " WHERE user_id = " + str(user_id)   
            select_query += " AND date >= " + str(from_date_start)
            select_query += " AND date <= " + str(to_date_end)    
            select_query += " ORDER BY DATE ASC"

            cursor.execute(select_query)
            response = cursor.fetchall()
            portfolio_history_df = general_utils.sql_to_dataframe(response, cursor)                                              
        except Exception as e:
            print("====================" + str(e) + "====================")
            portfolio_history_df = None
        finally:
            cursor.close()
            mydb.close()
            return portfolio_history_df
    
    def get_user_asset(self, user_id, assetId):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT * "
            select_query += " FROM user_assets"
            select_query += " WHERE user_id = " + str(user_id)
            select_query += " AND asset_id = " + str(assetId)

            cursor.execute(select_query)
            response = cursor.fetchall()
            asset_df = general_utils.sql_to_dataframe(response, cursor)                                         
        except Exception as e:
            print("====================" + str(e) + "====================")
            asset_df = None
        finally:
            cursor.close()
            mydb.close()
            return asset_df
    
    def get_user_assets(self, user_id):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT * "
            select_query += " FROM user_assets"
            select_query += " WHERE user_id = " + str(user_id)            

            cursor.execute(select_query)
            response = cursor.fetchall()
            asset_df = general_utils.sql_to_dataframe(response, cursor)                                                  
        except Exception as e:
            print("====================" + str(e) + "====================")
            asset_df = None
        finally:
            cursor.close()
            mydb.close()
            return asset_df

    def get_user_assets_history(self, user_id, date):
        '''
            Epistrefei tis metoxes pou eixe agorasmenes th sugekrimenh hmeromhnia enas user
        '''
        try:
            start, end = general_utils.getStartEndFromDate(date)
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT * "
            select_query += " FROM user_assets_history"
            select_query += " WHERE user_id = " + str(user_id)
            select_query += " AND date >= " + str(start) + " AND date <= " + str(end)

            cursor.execute(select_query)
            response = cursor.fetchall()
            asset_df = general_utils.sql_to_dataframe(response, cursor)                                           
        except Exception as e:
            print("====================" + str(e) + "====================")
            asset_df = None
        finally:
            cursor.close()
            mydb.close()
            return asset_df

    def get_sentiment_change(self, user_id, assets_type):
        connection_table = ""
        if assets_type == "watchlist":
            connection_table = "user_watchlist"
        elif assets_type == "investment":
            connection_table = "user_assets"
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT DISTINCT assets.name, sc1.sentiment_change_id, sc1.asset_id, sc1.date, s1.sentiment_id AS curr_id, s1.sentiment_text AS curr, s2.sentiment_id AS prev_id, s2.sentiment_text AS prev, sc1.probability"
            select_query +=" FROM sentiment_change AS sc1"
            select_query +=" INNER JOIN assets ON assets.asset_id = sc1.asset_id"
            select_query +=" INNER JOIN sentiments s1 ON sc1.current_sentiment = s1.sentiment_id"
            select_query +=" INNER JOIN sentiments s2 ON sc1.previous_sentiment = s2.sentiment_id"
            select_query +=" INNER JOIN "+ str(connection_table)+" ON sc1.asset_id = "+ str(connection_table)+".asset_id AND "+ str(connection_table)+".user_id = " + str(user_id)
            select_query +=" WHERE sc1.date = (SELECT MAX(sc2.date) FROM sentiment_change sc2 WHERE sc1.asset_id = sc2.asset_id)"
            cursor.execute(select_query)
            response = cursor.fetchall()
            result = general_utils.sql_to_json(response, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return result
   
    def get_reports_sentiment(self, user_id, assets_type):
        connection_table = ""
        if assets_type == "watchlist":
            connection_table = "user_watchlist"
        elif assets_type == "investment":
            connection_table = "user_assets"
        
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT  assets.name , reports.title, reports.full_text, reports.sentiment, reports.create_date AS date,"
            select_query += " writers_reliability.name AS writer_name, writers_reliability.reliability AS writer_reliability,"
            select_query += " source_reliability.name AS source_name, source_reliability.reliability AS source_reliability"
            select_query += " FROM reports" 
            select_query += " INNER JOIN "+ str(connection_table) +" ON "+ str(connection_table) +".asset_id = reports.asset_id AND "+ str(connection_table) +".user_id = " + str(user_id)
            select_query += " INNER JOIN assets ON assets.asset_id = reports.asset_id"
            select_query += " INNER JOIN writers_reliability ON writers_reliability.writers_reliability_id = reports.writer"
            select_query += " INNER JOIN source_reliability ON source_reliability.source_reliability_id = reports.source"
            select_query += " WHERE reports.source < 6" 
            select_query += " GROUP BY reports.full_text"
            cursor.execute(select_query)
            response = cursor.fetchall()
            result = general_utils.sql_to_json(response,cursor)

        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return result

    def get_active_signals_by_user_id(self, user_id):
        result = {'investment_signals' : [], 'watchlist_signals': [], 'investment_assets': [], 'watchlist_assets': []}
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT DISTINCT s1.signal_id, s1.asset_id, s1.probability, s1.date, signal_type.signal_type_id, signal_type.type, signal_action.signal_action_id, signal_action.action, assets.name, assets.yahoo_ticker FROM signals s1"
            select_query += " INNER JOIN user_assets"
            select_query += " ON user_assets.asset_id = s1.asset_id"
            select_query += " INNER JOIN assets"
            select_query += " ON user_assets.asset_id = assets.asset_id"
            select_query += " INNER JOIN signal_type"
            select_query += " ON signal_type.signal_type_id = s1.type"
            select_query += " INNER JOIN signal_action"
            select_query += " ON signal_action.signal_action_id = s1.action"
            select_query += " WHERE s1.date = (SELECT MAX(s2.date) FROM signals s2 WHERE s1.asset_id = s2.asset_id AND s1.type = s2.type)"
            select_query += " AND s1.signal_id NOT IN (SELECT DISTINCT signal_id FROM user_assets WHERE user_assets.user_id = " + str(user_id)
            select_query += " AND signal_id IS NOT NULL )"
            select_query += " AND user_assets.user_id = " + str(user_id)
            cursor.execute(select_query)
            response = cursor.fetchall()
            result["investment_signals"] = general_utils.sql_to_json(response, cursor)

            select_query2 =  " SELECT DISTINCT s1.signal_id, s1.asset_id, s1.probability, s1.date, signal_type.signal_type_id, signal_type.type, signal_action.signal_action_id, signal_action.action, assets.name, assets.yahoo_ticker FROM signals s1"
            select_query2 += " INNER JOIN user_watchlist"
            select_query2 += " ON user_watchlist.asset_id = s1.asset_id"
            select_query2 += " INNER JOIN assets"
            select_query2 += " ON user_watchlist.asset_id = assets.asset_id"
            select_query2 += " INNER JOIN signal_type"
            select_query2 += " ON signal_type.signal_type_id = s1.type"
            select_query2 += " INNER JOIN signal_action"
            select_query2 += " ON signal_action.signal_action_id = s1.action"
            select_query2 += " WHERE s1.date = (SELECT MAX(s2.date) FROM signals s2 WHERE s1.asset_id = s2.asset_id AND s1.type = s2.type)"
            select_query2 += " AND s1.signal_id NOT IN (SELECT DISTINCT signal_id FROM user_assets WHERE user_assets.user_id = " + str(user_id)
            select_query2 += " AND signal_id IS NOT NULL )"
            select_query2 += " AND s1.action != 2"
            select_query2 += " AND user_watchlist.user_id = " + str(user_id)
            cursor.execute(select_query2)
            response = cursor.fetchall()
            result["watchlist_signals"] = general_utils.sql_to_json(response, cursor)

            select_query3 = "SELECT assets.yahoo_ticker FROM user_assets"
            select_query3 += " INNER JOIN assets"
            select_query3 += " ON user_assets.asset_id = assets.asset_id"
            select_query3 += " WHERE user_id = " + str(user_id) 
            cursor.execute(select_query3)
            response3 = cursor.fetchall()
            for d in response3:
                result["investment_assets"].append(d[0])
            
            select_query4 = "SELECT assets.yahoo_ticker FROM user_watchlist"
            select_query4 += " INNER JOIN assets"
            select_query4 += " ON user_watchlist.asset_id = assets.asset_id"
            select_query4 += " WHERE user_id = " + str(user_id) 
            cursor.execute(select_query4)
            response4 = cursor.fetchall()
            for d in response4:
                result["watchlist_assets"].append(d[0])

        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result   


    def get_signals(self, user_id):
        result = {'investment_signal' : [], 'watchlist_signal': [], 'investment_assets': [], 'watchlist_assets': []}
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()

            select_query = " SELECT SUM(user_assets.percentage), s1.action, s1.type, s1.probability, assets.*, s1.date"
            select_query +=" FROM user"
            select_query +=" INNER JOIN user_assets"
            select_query +=" ON user.user_id = user_assets.user_id AND user.user_id = " + str(user_id)            
            select_query +=" INNER JOIN assets"
            select_query +=" ON user_assets.asset_id = assets.asset_id"
            select_query +=" INNER JOIN signals s1"
            select_query +=" ON user_assets.asset_id = s1.asset_id"
            select_query +=" WHERE s1.date = (SELECT MAX(s2.date) FROM signals s2 WHERE s1.asset_id = s2.asset_id AND s1.type = s2.type)"
            select_query +=" GROUP BY assets.asset_id, s1.type"
            select_query +=" ORDER BY s1.date DESC"
            cursor.execute(select_query)
            response = cursor.fetchall()
            test = general_utils.sql_to_json(response, cursor)
            for d in response:
                result["investment_signal"].append({
                    "percentage": d[0], 
                    "signal": d[1], 
                    "type": d[2], 
                    "probability": d[3],
                    "asset": {
                        "asset_id": d[4],
                        "name": d[5],
                        "ticker": d[6],
                        "description": d[7],
                        "class": d[8],
                        "yahoo_ticker": d[9]
                    },
                    "date": d[10]
                    })

            select_query2 = "SELECT s1.action, s1.type, s1.probability, assets.*, s1.date"
            select_query2 +=" FROM user"
            select_query2 +=" INNER JOIN user_watchlist"
            select_query2 +=" ON user.user_id = user_watchlist.user_id AND user.user_id = " + str(user_id)            
            select_query2 +=" INNER JOIN assets"
            select_query2 +=" ON user_watchlist.asset_id = assets.asset_id"
            select_query2 +=" INNER JOIN signals s1"
            select_query2 +=" ON user_watchlist.asset_id = s1.asset_id"
            select_query2 +=" WHERE s1.signal = 'buy' AND s1.date = (SELECT MAX(s2.date) FROM signals s2 WHERE s1.asset_id = s2.asset_id AND s1.type = s2.type)"
            select_query2 +=" ORDER BY s1.date DESC"
            cursor.execute(select_query2)
            response2 = cursor.fetchall()
            watchlist = []
            for d in response2:
                result["watchlist_signal"].append({
                    "signal": d[0], 
                    "type": d[1], 
                    "probability": d[2],
                    "asset": {
                        "asset_id": d[3],
                        "name": d[4],
                        "ticker": d[5],
                        "description": d[6],
                        "class": d[7],
                        "yahoo_ticker": d[8]
                    },
                    "date": d[9]
                    })

            select_query3 = "SELECT assets.yahoo_ticker FROM user_assets"
            select_query3 += " INNER JOIN assets"
            select_query3 += " ON user_assets.asset_id = assets.asset_id"
            select_query3 += " WHERE user_id = " + str(user_id) 
            cursor.execute(select_query3)
            response3 = cursor.fetchall()
            for d in response3:
                result["investment_assets"].append(d[0])
            
            select_query4 = "SELECT assets.yahoo_ticker FROM user_watchlist"
            select_query4 += " INNER JOIN assets"
            select_query4 += " ON user_watchlist.asset_id = assets.asset_id"
            select_query4 += " WHERE user_id = " + str(user_id) 
            cursor.execute(select_query4)
            response4 = cursor.fetchall()
            for d in response4:
                result["watchlist_assets"].append(d[0])
            
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return result            

    def get_all_signals(self, signal_types=None):

        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT assets.name as 'Asset', signals.action as 'Signal', signals.date as 'Date', signals.type as 'Type'"
            select_query += " FROM signals"
            select_query += " INNER JOIN assets"
            select_query += " ON signals.asset_id = assets.asset_id"
            if signal_types is not None:
                select_query += " WHERE"
                for idx, type in enumerate(signal_types):
                    select_query += " signals.type =" + str(type)
                    if idx+1 < len(signal_types):
                        select_query += " OR"
            cursor.execute(select_query)
            response = cursor.fetchall()
            result = general_utils.sql_to_json(response, cursor)

        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return result

    def get_signals_acceptance_history(self, userId):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT signals_acceptance_history.*, assets.name, signal_action.action, signal_type.type"
            select_query += " FROM signals_acceptance_history"
            select_query += " INNER JOIN signals "
            select_query += " ON signals.signal_id = signals_acceptance_history.signal_id"
            select_query += " INNER JOIN signal_type "
            select_query += " ON signal_type.signal_type_id = signals.type"
            select_query += " INNER JOIN signal_action "
            select_query += " ON signal_action.signal_action_id = signals.action"
            select_query += " INNER JOIN assets "
            select_query += " ON assets.asset_id = signals.asset_id"
            select_query += " WHERE user_id = " + str(userId)

            cursor.execute(select_query)
            response = cursor.fetchall()
            signals_df = general_utils.sql_to_dataframe(response, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
            signals_df = None
        finally:
            cursor.close()
            mydb.close()
            return signals_df

    def get_asset_indices(self, assetId, from_date, to_date):
        try:
            result = []
            resp = {}
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT asset_indeces.high, asset_indeces.low, asset_indeces.open, asset_indeces.close, asset_indeces.volume, asset_indeces.date, assets.name"
            select_query += " FROM asset_indeces"
            select_query += " INNER JOIN assets"
            select_query += " ON assets.asset_id = asset_indeces.asset_id AND asset_indeces.asset_id =" + str(assetId)
            if from_date is not None:
                select_query += " AND asset_indeces.date >=" + str(from_date)
            if to_date is not None:
                select_query += " AND asset_indeces.date <=" + str(to_date)
            select_query += " ORDER BY asset_indeces.date"            
            cursor.execute(select_query)
            all_entries = cursor.fetchall()
            if len(all_entries) > 0:      
                resp = {"asset_name": all_entries[0][6]}
                for entry in all_entries:
                    result.append({"high": entry[0], 
                                    "low": entry[1], 
                                    "open": entry[2],
                                    "close": entry[3],
                                    "volume": entry[4],
                                    "date": entry[5]})
                resp["asset_indices"] = result
            else: 
                resp["asset_indices"] = []
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_asset_sentiment(self, assetId, from_date, to_date):
        try:
            result = []
            resp = {}
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT asset_sentiment.sentiment, sentiments.sentiment_text, asset_sentiment.date, assets.name"
            select_query += " FROM asset_sentiment"
            select_query += " INNER JOIN assets"
            select_query += " ON assets.asset_id = asset_sentiment.asset_id AND asset_sentiment.asset_id =" + str(assetId) +  " AND asset_sentiment.date >="  + str(from_date) +" AND asset_sentiment.date <=" +str(to_date)
            select_query += " INNER JOIN sentiments ON asset_sentiment.sentiment = sentiments.sentiment_id"
            select_query += " ORDER BY asset_sentiment.date DESC LIMIT 1"            
            cursor.execute(select_query)
            all_entries = cursor.fetchall()
            result = general_utils.sql_to_json(all_entries, cursor)
            if len(result)>0:
                resp = result[0]  
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_latest_indeces_date(self):
        try:
            resp = {}
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT asset_indeces.date"
            select_query += " FROM asset_indeces"
            select_query += " ORDER BY asset_indeces.date DESC"
            cursor.execute(select_query)
            all_entries = cursor.fetchall()
            if len(all_entries) > 0:
                resp = {"latest_date": all_entries[0]}
            else:
                resp["latest_date"] = []
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_asset_by_id(self, assetId):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT * "
            select_query += " FROM assets"            
            select_query += " WHERE asset_id =" + str(assetId)

            cursor.execute(select_query)
            response = cursor.fetchall()
            
            result = general_utils.sql_to_json(response, cursor)
            if len(result)>0:
                resp = result[0]  
        except Exception as e:
            resp = None
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_asset_by_name(self, assetName):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT * "
            select_query += " FROM assets"
            select_query += " WHERE name =" + "'" + assetName + "'"

            cursor.execute(select_query)
            response = cursor.fetchall()

            result = general_utils.sql_to_json(response, cursor)
            if len(result) > 0:
                resp = result[0]
        except Exception as e:
            resp = None
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return resp
    
    def get_asset_by_ticker(self, assetTicker):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT * "
            select_query += " FROM assets"
            select_query += " WHERE ticker =" + "'" + assetTicker + "'"

            cursor.execute(select_query)
            response = cursor.fetchall()

            result = general_utils.sql_to_json(response, cursor)
            if len(result) > 0:
                resp = result[0]
        except Exception as e:
            resp = None
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return resp

    def remove_duplicate_tweets(self, tweet_list, platform):
        new_tweet_list = []
        try:
            source = self.get_source_reliability_id_by_name(platform)
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            cursor.execute("SELECT source_id FROM reports where source = " + str(source) + " and asset_id = " + str(tweet_list[0]["asset_id"]))
            results = cursor.fetchall()
            source_ids = []
            for source_id in results:
                source_ids.append(source_id[0])
            if len(source_ids) > 0:
                indices_for_removal = []
                for i, tweet in enumerate(tweet_list):
                    if str(tweet["tweet_id"]) in source_ids:
                        indices_for_removal.append(i)
                new_tweet_list = [i for j, i in enumerate(tweet_list) if j not in indices_for_removal]
            else:
                new_tweet_list = tweet_list

            # last_tweet_in_db = results[0][0]
            # if last_tweet_in_db is not None:
            #     indices_for_removal = []
            #     for i, tweet in enumerate(tweet_list):
            #         if int(tweet["tweet_id"]) <= int(last_tweet_in_db):
            #             indices_for_removal.append(i)
            #     new_tweet_list = [i for j, i in enumerate(tweet_list) if j not in indices_for_removal]
            # else:
            #     new_tweet_list = tweet_list
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return new_tweet_list

    def get_latest_article_date(self):
        latest_date = 0
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            cursor.execute("SELECT retrieve_date FROM aspendys.reports where source_id = 'article'")
            results = cursor.fetchall()
            # TO_BE_IMPROVED
            if len(results) == 0:
                today = datetime.today()
                yesterday = today - timedelta(days=1)
                latest_date = yesterday
            else:
                date_list = []
                for result in results:
                    date = datetime.fromtimestamp(result[0] / 1000.0)
                    date_list.append(date)
                latest_date = max(d for d in date_list)
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return latest_date

    def insert_tweets(self, new_tweet_list):
        success = False
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            for tweet in new_tweet_list:
                source_id = self.get_source_reliability_id_by_name(tweet["platform"])
                writer_id = self.get_writer_reliability_id_by_name(tweet["user"])
                sql = "INSERT INTO reports (source_id, asset_id, title, full_text, create_date, retrieve_date, " 
                sql += "sentiment, source, writer, stocktwits_sentiment, " 
                sql += " sentiment_is_processed, source_reliability_is_processed, writer_reliability_is_processed, " 
                sql += "report_reliability, metadata)" 
                sql += " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                values = (tweet["tweet_id"], tweet["asset_id"], ' '.join(tweet["text"].split()[:5])+"...", tweet["text"], tweet["create_date"], tweet["retrieve_date"], 
                        None, source_id, writer_id, tweet["sentiment"], 
                        False, False, False, 
                        0, json.dumps(tweet["metadata"]))
                cursor.execute(sql, values)
            mydb.commit()
            success = True
        except Exception as e:
            success = False
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close() 
            return success 

    def insert_news(self, article_list):
        success = False
        try:
            article_list.reverse()
            new_article_list = article_list
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            for article_pair in new_article_list:
                article = article_pair[0]
                asset_id = article_pair[1]
                source_id = self.get_source_reliability_id_by_name(article["source"]['name']) if article["source"]['name'] != None else None
                writer_id = self.get_writer_reliability_id_by_name(article["author"]) if article["author"] != None else None
                create_date = calendar.timegm(datetime.strptime(article["publishedAt"],'%Y-%m-%dT%H:%M:%SZ').timetuple())*1000
                retrieve_date = general_utils.getCurrentTimestsamp()
                sql = "INSERT INTO reports (source_id, asset_id, title, full_text, create_date, retrieve_date, sentiment, source, writer, " 
                sql +="stocktwits_sentiment, sentiment_is_processed, source_reliability_is_processed, writer_reliability_is_processed, report_reliability, metadata) "
                sql +=" VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                values = ('article', asset_id, article['title'], article["content"], create_date, retrieve_date, None, source_id, writer_id, 
                          None, False, False, False, 1, None)
                cursor.execute(sql, values)
            
            sql2 = "DELETE FROM reports WHERE full_text LIKE '%Completing the CAPTCHA%'"
            cursor.execute(sql2)

            sql3 = 'DELETE FROM reports WHERE full_text LIKE "%If you typed the URL into your browser, check that you entered it correctly%"'
            cursor.execute(sql3)

            sql4 = 'DELETE FROM reports WHERE full_text LIKE "%Please make sure your browser supports JavaScript and cookies and that you are not blocking them from loading%"'
            cursor.execute(sql4)

            mydb.commit()
            success = True
        except Exception as e:
            success = False
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()   
            return success 

    def get_user_portfolio(self, user_id):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT * "
            select_query += " FROM portfolio"
            select_query += " WHERE user_id = " + str(user_id)            

            cursor.execute(select_query)
            response = cursor.fetchall()
            asset_df = general_utils.sql_to_dataframe(response, cursor)                                        
        except Exception as e:
            print("====================" + str(e) + "====================")
            asset_df = None
        finally:
            cursor.close()
            mydb.close()
            return asset_df

    def update_user_portfolio(self, user_id, profile_type, optimization_method, cash, investment_percentage, invested_money, portfolio_assets_value):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            update_query =  " UPDATE portfolio "
            update_query += " SET profile_type = " + str(profile_type)
            update_query += ", optimization_method = " + str(optimization_method)
            update_query += ", cash = " + str(cash)
            update_query += ", investment_percentage = " + str(investment_percentage)
            update_query += ", invested_money = " + str(invested_money)
            update_query += ", portfolio_assets_value = " + str(portfolio_assets_value)            
            update_query += " WHERE user_id = " + str(user_id)
            cursor.execute(update_query)    

            mydb.commit()        
            resp = { "status": "SUCCESS" }
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" }
        finally:
            cursor.close()
            mydb.close()
            return resp
    
    def insert_signals(self, signals_list):
        success = False
        result = []
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)
            records_to_insert = []
            for item in signals_list:
                cursor.execute("SELECT * FROM assets WHERE yahoo_ticker='"+item["yahoo_ticker"]+"'")
                query_result = cursor.fetchall()
                asset_data = general_utils.sql_to_json(query_result, cursor)
                if len(asset_data)>0:
                    signal = asset_data[0]
                    signal["probability"] = item["probability"]
                    signal["date"] = item["date"]
                    
                    cursor.execute("SELECT * FROM signal_action WHERE signal_action_id = " + str(item["action"]))
                    action_result = cursor.fetchall()
                    signal_action = general_utils.sql_to_json(action_result, cursor)
                    if len(signal_action)>0:
                        signal["action"] = signal_action[0]["action"]
                        signal["signal_action_id"] = signal_action[0]["signal_action_id"]

                    cursor.execute("SELECT * FROM signal_type WHERE signal_type_id = " + str(item["type"]))
                    type_result = cursor.fetchall()
                    signal_type = general_utils.sql_to_json(type_result, cursor)
                    if len(signal_type)>0:
                        signal["type"] = signal_type[0]["type"]
                        signal["signal_type_id"] = signal_type[0]["signal_type_id"]

                    insert_query = """INSERT INTO signals (`asset_id`, `probability`, `action`, `type`, `date`) VALUES (%s, %s, %s, %s, %s) """
                    values = (asset_data[0]["asset_id"], item["probability"], int(item["action"]), item["type"], item["date"])
                    cursor.execute(insert_query, values)
                    mydb.commit()
                    
                    cursor.execute("SELECT LAST_INSERT_ID()")
                    id_result = cursor.fetchone()
                    if len(id_result)>0:
                        signal["signal_id"] = id_result[0]
                    
                    result.append(signal)
                     
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result

    def insert_signal(self, asset_id, probability, action, signal_type, date):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()            
            query = "INSERT INTO signals (asset_id, probability, action, type, date) VALUES (%s, %s, %s, %s, %s)"
            values = (asset_id, probability, action, signal_type, date)
            cursor.execute(query, values)
            mydb.commit()
            resp = { "status": "SUCCESS" } 
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp

    def update_history_with_new_action(self, user_id, action, amount, asset_id, date, asset_value):
        '''
            user_id: user_id
            action: 1->CASH, 2->BUY, 3->SELL
            amount: if CASH then CASH value, if (BUY of SELL) then amount of asset
            asset_id: asset_id
            date: date
            asset_value: if CASH then None, if (BUY of SELL) then BUY or SELL value
        '''
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()            
            sql = "INSERT INTO history (user_id, action, amount, asset_id, date, asset_value)" \
                " VALUES (%s, %s, %s, %s, %s, %s)"
            values = (user_id, action, str(amount), asset_id, date, asset_value)
            cursor.execute(sql, values)
            mydb.commit()
            resp = { "status": "SUCCESS" } 
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp

    def update_user_assets_history(self, user_id):
        try:

            resp = {}
            date_start, date_end = general_utils.getTodayStartEndTimestamps()

            today_user_assets_history_df = self.get_user_assets_history(user_id, date_start)
            today_user_assets_df = self.get_user_assets(user_id)

            today_user_assets_history = today_user_assets_history_df.to_dict("records")
            today_user_assets = today_user_assets_df.to_dict("records")
            
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()

            for user_asset in today_user_assets:
                common = None
                for user_asset_history in today_user_assets_history:
                    if user_asset["user_id"] == user_asset_history["user_id"] and user_asset["asset_id"] == user_asset_history["asset_id"] and user_asset["amount"] == user_asset_history["amount"]:
                        common = user_asset_history
                        break
                if common != None:
                    today_user_assets_history.remove(common)
                    update_query =  " UPDATE user_assets_history "
                    update_query += " SET percentage = " + str(user_asset["percentage"])
                    update_query += " WHERE user_id = " + str(user_id)
                    update_query += " AND asset_id = " + str(user_asset["asset_id"])
                    update_query += " AND amount = " + str(user_asset["amount"])
                    update_query += " AND date >= " + str(date_start)
                    update_query += " AND date <= " + str(date_end)
                    cursor.execute(update_query)
                else:
                    sql = "INSERT INTO user_assets_history (user_id, asset_id, percentage, date, amount, asset_buy_value) VALUES (%s, %s, %s, %s, %s, %s)"
                    values = (int(user_asset["user_id"]), int(user_asset["asset_id"]), float(user_asset["percentage"]), int(date_start), int(user_asset["amount"]), float(user_asset["asset_buy_value"]))
                    cursor.execute(sql, values)
        
            mydb.commit()
            resp = { "status": "SUCCESS" } 
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp

    def update_portfolio_history(self, user_id):
        try:       
            today_timestamp = general_utils.getCurrentTimestsamp()
            date_start, date_end = general_utils.getStartEndFromDate(today_timestamp)

            resp = {}
            portfolio = self.get_user_portfolio(user_id)
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()

            #des an uparxei katagrafh
            cursor.execute("SELECT * FROM portfolio_history WHERE user_id = \'" + str(user_id) + "'" + " AND date >=" + str(date_start) + " AND date <=" + str(date_end))
            portfolio_history = cursor.fetchall()
            row = portfolio.iloc[0]
            if len(portfolio_history) == 0:
                # an den uparxei grapse kainourgio entry
                sql = "INSERT INTO portfolio_history (user_id, date, profile_type, optimization_method, cash, investment_percentage, invested_money, portfolio_assets_daily_value)" \
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                values = (user_id, date_start, int(row["profile_type"]), int(row["optimization_method"]), float(row["cash"]), float(row["investment_percentage"]), float(row["invested_money"]), float(row["portfolio_assets_value"]))
                cursor.execute(sql, values)
            else:
                # an uparxei kane update to hdh uparxon
                update_query =  " UPDATE portfolio_history "
                update_query += " SET cash = " + str(row["cash"])
                update_query += " , investment_percentage = " + str(row["investment_percentage"])
                update_query += " , invested_money = " + str(row["invested_money"])
                update_query += " , portfolio_assets_daily_value = " + str(row["portfolio_assets_value"])
                update_query += " WHERE user_id = " + str(user_id)
                update_query += " AND date >=" + str(date_start)
                update_query += " AND date <=" + str(date_end)
                cursor.execute(update_query)                  

            mydb.commit()
            resp = { "status": "SUCCESS" } 
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp

    def update_user_assets_percentages(self, user_id, user_assets):
        try:    
            today_timestamp = general_utils.getCurrentTimestsamp()
            date_start, date_end = general_utils.getStartEndFromDate(today_timestamp)

            resp = {}
            portfolio = self.get_user_portfolio(user_id)
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            for index, row in user_assets.iterrows():
                update_query =  " UPDATE user_assets "
                update_query += " SET percentage = " + str(row["percentage"])
                update_query += " WHERE user_id = " + str(user_id)
                update_query += " AND asset_id = " + str(row["asset_id"])
                update_query += " AND amount = " + str(row["amount"])
                cursor.execute(update_query)                  
                mydb.commit()

            for index, row in user_assets.iterrows():
                update_query =  " UPDATE user_assets_history "
                update_query += " SET percentage = " + str(row["percentage"])
                update_query += " WHERE user_id = " + str(user_id)
                update_query += " AND asset_id = " + str(row["asset_id"])
                update_query += " AND amount = " + str(row["amount"])
                update_query += " AND date >= " + str(date_start)
                update_query += " AND date <= " + str(date_end)
                cursor.execute(update_query)                  
                mydb.commit()

            resp = { "status": "SUCCESS" }
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" }
        finally:
            cursor.close()
            mydb.close()
            return resp
    
    def insert_accepted_signal_to_signals_history(self, user_id, signal_id, acceptance_date, asset_buy_value, sell_date=None, success=None): 
        try:
            resp = {}            
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            sql1 = "INSERT INTO signals_acceptance_history (user_id, signal_id, acceptance_date, asset_buy_value, sell_date, success) VALUES (%s, %s, %s, %s, %s, %s)"
            values1 = (user_id, signal_id, acceptance_date, asset_buy_value, sell_date, success)
            cursor.execute(sql1, values1)    

            mydb.commit()
            resp = { "status": "SUCCESS" } 
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp

    def update_db_during_buy_asset(self, user_id, data):
        try:
            date_start, date_end = general_utils.getStartEndFromDate(data["date"])
            resp = {}
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()

            sql1 =  " SELECT * FROM portfolio_history WHERE user_id = " + str(user_id) + " AND date >=" + str(date_start) + " AND date <=" + str(date_end) 
            cursor.execute(sql1)
            portfolio_history = cursor.fetchall()
            if len(portfolio_history) == 0:
                #insert
                sql2 =  " SELECT * FROM portfolio WHERE user_id = " + str(user_id)
                cursor.execute(sql2)
                response = cursor.fetchall()
                user_portfolio = general_utils.sql_to_dataframe(response, cursor, True)
                                
                portfolio_history_data = (user_id, date_start, int(user_portfolio["profile_type"]), int(user_portfolio["optimization_method"]), float(user_portfolio["cash"]), float(user_portfolio["investment_percentage"]), float(user_portfolio["invested_money"]), float(user_portfolio["portfolio_assets_value"]))
                cursor.execute("INSERT INTO portfolio_history (user_id, date, profile_type, optimization_method, cash, investment_percentage, invested_money, portfolio_assets_daily_value) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", portfolio_history_data)
                mydb.commit()


            sql3 = "INSERT INTO user_assets (user_id, asset_id, percentage, date, amount, asset_buy_value, signal_id) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            values3 = (user_id, data["assetId"], None, data["date"], data["amount"], data["asset_buy_value"], data["signal_id"])
            
            sql4 = "INSERT INTO user_assets_history (user_id, asset_id, percentage, date, amount, asset_buy_value, signal_id) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            values4 = (user_id, data["assetId"], None, date_start, data["amount"], data["asset_buy_value"], data["signal_id"])

            sql5 =  " UPDATE portfolio SET cash = " + str(data["cash"]) +" WHERE user_id = " + str(user_id)

            sql6 =  " UPDATE portfolio_history SET cash = " + str(data["cash"]) +" WHERE user_id = " + str(user_id) + " AND date >=" + str(date_start) + " AND date <=" + str(date_end) 

            sql7 = "INSERT INTO history (user_id, action, amount, asset_id, date, asset_value) VALUES (%s, %s, %s, %s, %s, %s)"
            values7 = (user_id, data["action"], data["amount"], data["assetId"], data["date"], data["asset_buy_value"])
            
            sql8 = "DELETE FROM user_watchlist WHERE asset_id = " + str(data["assetId"])
            
            cursor.execute(sql3, values3)
            cursor.execute(sql4, values4)
            cursor.execute(sql5)
            cursor.execute(sql6)
            cursor.execute(sql7, values7)
            cursor.execute(sql8)
            mydb.commit()
            resp = { "status": "SUCCESS" }
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" }
        finally:
            cursor.close()
            mydb.close()
            return resp

    def update_db_during_sell_asset(self, user_id, data):
        try:
            date_start, date_end = general_utils.getStartEndFromDate(data["date"])
            resp = {}
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()            

            # kanw elegxo gia to an uparxei eggrafh gia th shmerinh mera sto portfolio_history 
            # an oxi prepei na ginei INSERT me vash ta data tou portfolio
            sql6 =  " SELECT * FROM portfolio_history WHERE user_id = " + str(user_id) + " AND date >=" + str(date_start) + " AND date <=" + str(date_end) 
            cursor.execute(sql6)
            portfolio_history = cursor.fetchall()
            if portfolio_history == None:
                #insert
                sql7 =  " SELECT * FROM portfolio WHERE user_id = " + str(user_id)
                cursor.execute(sql7)
                response = cursor.fetchall()

                user_portfolio = general_utils.sql_to_dataframe(response, cursor, True) 
                
                
                portfolio_history_data = (user_id, date_start, user_portfolio["profile_type"], user_portfolio["optimization_method"], user_portfolio["cash"], user_portfolio["investment_percentage"], user_portfolio["invested_money"], user_portfolio["portfolio_assets_value"])
                cursor.execute("INSERT INTO portfolio_history (user_id, date, profile_type, optimization_method, cash, investment_percentage, invested_money, portfolio_assets_daily_value) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", portfolio_history_data)
                mydb.commit()

            sql1 =  " DELETE FROM user_assets WHERE user_id = " + str(user_id) + " AND asset_id = " + str(data["assetId"])
            sql2 =  " DELETE FROM user_assets_history WHERE user_id = " + str(user_id) + " AND asset_id = " + str(data["assetId"]) + " AND date >=" + str(date_start) + " AND date <=" + str(date_end) 
            sql3 =  " UPDATE portfolio SET cash = " + str(data["new_cash_value"]) +" WHERE user_id = " + str(user_id)
            sql4 =  " UPDATE portfolio_history SET cash = " + str(data["new_cash_value"]) +" WHERE user_id = " + str(user_id) + " AND date >=" + str(date_start) + " AND date <=" + str(date_end) 

            sql5 = "INSERT INTO history (user_id, action, amount, asset_id, date, asset_value) VALUES (%s, %s, %s, %s, %s, %s)"
            values5 = (user_id, data["action"], float(data["amount"]), data["assetId"], data["date"], float(data["asset_value"]))
            
            cursor.execute(sql1)
            cursor.execute(sql2)
            cursor.execute(sql3)
            cursor.execute(sql4)
            cursor.execute(sql5, values5)
            
            mydb.commit()
            resp = { "status": "SUCCESS" } 
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_all_users(self):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT * FROM user WHERE roles='investor'"
            cursor.execute(select_query)
            all_users = cursor.fetchall()
            for user in all_users:
                result.append({"user_id": user[0]})
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result

    def update_assets_and_portfolio_history_for_all_users(self):
        success = False
        try:
            all_users = self.get_all_users()
            for user in all_users:
                self.update_user_assets_history(user["user_id"])
                self.update_portfolio_history(user["user_id"])
                print(str(user["user_id"]) + " | Assets and Porfolio History Updated")
            success = True
            print("-------- Update User Assets And Portfolio History--------")
        except Exception as e:
            print("====================" + str(e) + "====================")
            print("-------- PROBLEM: Update User Assets And Portfolio History NOT completed--------")
            success = False
                        
        finally:
            return success

    def add_watchlist_asset(self, user_id, assetId):
        success = False
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "INSERT INTO user_watchlist (user_id, asset_id) VALUES (%s, %s)"
            values = (int(user_id), int(assetId))
            cursor.execute(query, values)
            mydb.commit()
            success = True
        except Exception as e:
            print("====================" + str(e) + "====================")
            success = False
        finally:
            cursor.close()
            mydb.close()
            return success

    def delete_watchlist_asset(self, user_id, assetId):
        success = False
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "DELETE FROM user_watchlist WHERE user_id = " + user_id + " AND asset_id = " + assetId
            cursor.execute(query)    
            mydb.commit()      
            success = True
        except Exception as e:
            print("====================" + str(e) + "====================")
            success = False
        finally:
            cursor.close()
            mydb.close()
            return success


    def get_portfolio_register_date(self, user_id):
        date = 0        
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)
            query = "SELECT MIN(date) FROM portfolio_history WHERE user_id = " + str(user_id)
            cursor.execute(query)    
            result = cursor.fetchone()
            if len(result) > 0:
                date = int(result[0])
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return date

    def get_writer_reliability_id_by_name(self, writer_name):
        writers_reliability_id = -1        
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)
            writer_name = writer_name.replace("'", "’")
            select_query = "SELECT writers_reliability_id FROM writers_reliability WHERE name = '" + str(writer_name) + "'"
            cursor.execute(select_query)    
            result = cursor.fetchone()
            # print("=======================")
            # print(str(writer_name))
            # print(result)
            # print("=======================")
            if result == None:
                insert_query = "INSERT INTO writers_reliability (name) VALUES (%s)"
                insert_values = (str(writer_name),)
                cursor.execute(insert_query, insert_values)
                mydb.commit()
                
                cursor.execute(select_query)    
                result = cursor.fetchone()
                if len(result) > 0:
                    writers_reliability_id = result[0]
            else:
                writers_reliability_id = result[0]


        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return writers_reliability_id

    def get_source_reliability_id_by_name(self, source_name):
        source_reliability_id = -1        
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)
            select_query = "SELECT source_reliability_id FROM source_reliability WHERE name = '" + str(source_name) + "'"
            cursor.execute(select_query)    
            result = cursor.fetchone()
            if result == None:
                insert_query = "INSERT INTO source_reliability (name) VALUES (%s)"
                insert_values = (str(source_name),)
                cursor.execute(insert_query, insert_values)
                mydb.commit()
                
                cursor.execute(select_query)    
                result = cursor.fetchone()
                if len(result) > 0:
                    source_reliability_id = result[0]
            else:
                source_reliability_id = result[0]
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return source_reliability_id
			
			
    def update_signals_acceptance_history(self, user_id, data):
        try:
            resp = {}
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            for index, row in data.iterrows():
                update_query =  " UPDATE signals_acceptance_history "
                update_query += " SET sell_date = " + str(row["sell_date"])
                update_query += ", success = " + str(row["success"])
                update_query += ", asset_profit_percentage = " + str(row["asset_profit_percentage"])
                update_query += ", portfolio_asset_profit_percentage = " + str(row["portfolio_asset_profit_percentage"])
                update_query += ", asset_buy_value = " + str(row["buy_value"])
                update_query += ", asset_sell_value = " + str(row["sell_value"])
                update_query += " WHERE user_id = " + str(user_id)
                update_query += " AND signal_id = " + str(row["signal_id"])
                update_query += " AND acceptance_date = " + str(row["date"])
                cursor.execute(update_query)
            mydb.commit()
            resp = { "status": "SUCCESS" }
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" }
        finally:
            return resp
    
    def get_signals_acceptance_history_by_id(self, signals_acceptance_history_id):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query =  " SELECT signals_acceptance_history.*, assets.asset_id, assets.name, signals.action, signals.type"
            select_query += " FROM signals_acceptance_history"
            select_query += " INNER JOIN signals "
            select_query += " ON signals.signal_id = signals_acceptance_history.signal_id"
            select_query += " INNER JOIN assets "
            select_query += " ON assets.asset_id = signals.asset_id"
            select_query += " WHERE signals_acceptance_history_id = " + str(signals_acceptance_history_id)

            cursor.execute(select_query)
            response = cursor.fetchall()
            signals_df = general_utils.sql_to_dataframe(response, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
            signals_df = None
        finally:
            cursor.close()
            mydb.close()
            return signals_df


    def insert_asset_indices(self, asset, data, date):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()    
            asset_indeces_date = data["Date"].strftime('%Y-%m-%d')
            
            select_query = "SELECT * FROM asset_indeces" 
            select_query += " WHERE asset_id = " + str(asset["asset_id"])
            select_query += " AND date = " + str(data.Date.timestamp() * 1000)
            cursor.execute(select_query)
            entry = cursor.fetchall()
            if len(entry) == 0:
                insert_query = "INSERT INTO asset_indeces ("
                insert_query += "asset_id, high, low, open, close, volume, date )"
                insert_query += " VALUES ( %s, %s, %s, %s, %s, %s, %s)"
                insert_args = (str(asset["asset_id"]), 
                            str(data["High"]), 
                            str(data["Low"]), 
                            str(data["Open"]), 
                            str(data["Close"]), 
                            str(data["Volume"]), 
                            str(data.Date.timestamp() * 1000))
                cursor.execute(insert_query, insert_args)
                mydb.commit()  
                print(asset_indeces_date + " | " + asset["name"] + " | Asset Indices Updated")   
            else:
                print(asset_indeces_date + " | " + asset["name"] + " | Asset Indices Already exists for this date")                
        except Exception as e:
            print("==================== " + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()

    def get_unprocessed_reports(self):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "SELECT * FROM reports WHERE sentiment_is_processed = FALSE"   # sentiment_is_processed ?
            cursor.execute(query)
            result = cursor.fetchall()        
            reports_df = general_utils.sql_to_dataframe(result, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return reports_df    

    def get_assets_sentiment(self, from_date, to_date):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "SELECT * FROM asset_sentiment WHERE date >= "+ str(int(from_date)) + " AND date<=" + str(int(to_date))
            cursor.execute(query)
            result = cursor.fetchall()        
            sentiment_history_df = general_utils.sql_to_dataframe(result, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return sentiment_history_df

    def get_asset_indices_df(self, assetId, from_date, to_date):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "SELECT * FROM asset_indeces WHERE AND asset_id =" + str(assetId) +  " AND date >="  + str(from_date) +" AND date <=" +str(to_date)
            cursor.execute(query)
            result = cursor.fetchall()        
            asset_indices_df = general_utils.sql_to_dataframe(result, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return asset_indices_df

    def get_sentiment_change_for_backend(self, form_date, to_date):        
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "SELECT * FROM sentiment_change WHERE date>= "+ str(int(form_date)) + " AND date<=" + str(int(to_date))
            cursor.execute(query)
            response = cursor.fetchall()
            result = general_utils.sql_to_dataframe(response, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return result

    def update_report_sentiment_in_db(self, current_report_id, current_article_sentiment):
        try:
            resp = {}
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "UPDATE reports SET sentiment =%s, sentiment_is_processed =%s WHERE report_id =%s "
            cursor.execute(query, (int(current_article_sentiment), 1, int(current_report_id)))
            mydb.commit()
            resp = { "status": "SUCCESS" }
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" }
        finally:
            return resp  

    def insert_asset_sentiment(self, asset_id, sentiment, probability, date):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()            
            query = "INSERT INTO asset_sentiment (asset_id, sentiment, probability, date) VALUES (%s, %s, %s, %s)"
            values = (int(asset_id), int(sentiment), float(probability) ,int(date))
            cursor.execute(query, values)
            mydb.commit()
            resp = { "status": "SUCCESS" } 
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp

    def insert_sentiment_change(self, previous_sentiment, current_sentiment, current_date_in_millisecs, asset_id):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()            
            query = "INSERT INTO sentiment_change (previous_sentiment, current_sentiment, date, asset_id) VALUES (%s, %s, %s, %s)"
            values = (previous_sentiment, current_sentiment, current_date_in_millisecs, asset_id)
            cursor.execute(query, values)
            mydb.commit()
            resp = { "status": "SUCCESS" } 
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = { "status": "FAILED" } 
        finally:
            cursor.close()
            mydb.close()
            return resp

    def get_last_sentiment_for_asset(self, asset_id):
        result = -1
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)    
            query = "SELECT sentiment FROM asset_sentiment as1 WHERE asset_id=" + str(asset_id)
            query += " AND date = (SELECT MAX(date) FROM asset_sentiment as2 WHERE as2.asset_id = as1.asset_id)"
            cursor.execute(query)
            response = cursor.fetchone()
            if(len(response) > 0):
                result = response[0]
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result

    def get_last_probability_for_asset(self, asset_id):
        result = -1
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)    
            query = "SELECT probability FROM asset_sentiment as1 WHERE asset_id=" + str(asset_id)
            query += " AND date = (SELECT MAX(date) FROM asset_sentiment as2 WHERE as2.asset_id = as1.asset_id)"
            cursor.execute(query)
            response = cursor.fetchone()
            if(len(response) > 0):
                result = response[0]
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result
            

    def get_indeces_with_window_by_asset(self, asset_id, window):
        result = -1
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "SELECT close FROM asset_indeces WHERE asset_id= " + str(asset_id) + " ORDER BY date DESC LIMIT " + str(window)
            cursor.execute(query)
            response = cursor.fetchall()
            if(len(response) > 0):
                result = {"first": response[window - 1][0], "last": response[0][0]}
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result

    def get_indeces_history_by_asset(self, asset_id, window):
        result = {}
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "SELECT date, close FROM asset_indeces WHERE asset_id= " + str(asset_id) + " ORDER BY date DESC LIMIT " + str(window)
            cursor.execute(query)
            response = cursor.fetchall()
            indeces_history = general_utils.sql_to_dataframe(response, cursor)
            indeces_history = indeces_history.sort_values(by=['date'])
            result["label"] = indeces_history.apply(lambda row: general_utils.convertMillisecsToDate(row["date"]), axis=1).to_list()
            result["value"] = indeces_history["close"].to_list()
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result

            
    def get_signals_by_asset(self, asset_id, window):
        result = {"date": [], "type": [], "action": []}
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "SELECT action, type, date FROM signals WHERE asset_id= " + str(asset_id) + " ORDER BY date DESC LIMIT " + str(window)
            cursor.execute(query)
            response = cursor.fetchall()
            if len(response) > 0:
                signals = general_utils.sql_to_dataframe(response, cursor)
                signals = signals.sort_values(by=['date'])
                result["date"] = signals.apply(lambda row: general_utils.convertMillisecsToDate(row["date"]), axis=1).to_list()
                result["type"] = signals["type"].to_list()
                result["action"] = signals["action"].to_list()
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result

    def get_asset_info(self, asset_id):
        result = {"name": "", "currentValue": 0, "currentSentiment": ""}
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            # query = "SELECT action, type, date FROM signals WHERE asset_id= " + str(asset_id) + " ORDER BY date DESC LIMIT " + str(window)
            query = "SELECT name, asset_indeces.close AS currentValue, sentiments.sentiment_text AS currentSentiment FROM assets  "
            query += "INNER JOIN asset_indeces ON asset_indeces.asset_id = assets.asset_id "
            query += "INNER JOIN asset_sentiment ON asset_sentiment.asset_id = assets.asset_id "
            query += "INNER JOIN sentiments ON asset_sentiment.sentiment = sentiments.sentiment_id "
            query += "WHERE assets.asset_id= " + str(asset_id) + " "
            query += "ORDER BY asset_indeces.date DESC, asset_sentiment.date DESC LIMIT 1"
            cursor.execute(query)
            response = cursor.fetchall()
            result_tmp = general_utils.sql_to_json(response, cursor)
            if len(result_tmp)>0:
                result = result_tmp[0]
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result


    def insert_emergencies(self, emergencies):
        resp = {"status": "SUCCESS"}
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            for item in emergencies:
                query = "INSERT INTO emergencies (asset_id, alarm, date) VALUES (%s, %s, %s)"
                values = (item['asset_id'], item['alarm'], item['date'])
                cursor.execute(query, values)
                mydb.commit()
        except Exception as e:
            print("====================" + str(e) + "====================")
            resp = {"status": "FAILED"}
        finally:
            cursor.close()
            mydb.close()
            return resp


    def get_user_assets_sentiment(self, user_id):
        result = {}

        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = " SELECT assets.name, sentiments.sentiment_text AS sentiment,  asset_sentiment.probability FROM assets "
            select_query +=" INNER JOIN user_assets ON user_assets.asset_id = assets.asset_id "
            select_query +=" INNER JOIN asset_sentiment ON asset_sentiment.asset_id = assets.asset_id"
            select_query +=" INNER JOIN sentiments ON asset_sentiment.sentiment = sentiments.sentiment_id"
            select_query +=" WHERE asset_sentiment.date = (SELECT MAX(as2.date) FROM asset_sentiment as2 WHERE asset_sentiment.asset_id = as2.asset_id)"
            select_query +=" AND user_assets.user_id =" + str(user_id)
            select_query +=" GROUP BY user_assets.asset_id"
            
            cursor.execute(select_query)
            response = cursor.fetchall()
            result["investments_sentiment"] = general_utils.sql_to_json(response, cursor)

            select_query2 = " SELECT assets.name, sentiments.sentiment_text AS sentiment,  asset_sentiment.probability FROM assets "
            select_query2 +=" INNER JOIN user_watchlist ON user_watchlist.asset_id = assets.asset_id "
            select_query2 +=" INNER JOIN asset_sentiment ON asset_sentiment.asset_id = assets.asset_id"
            select_query2 +=" INNER JOIN sentiments ON asset_sentiment.sentiment = sentiments.sentiment_id"
            select_query2 +=" WHERE asset_sentiment.date = (SELECT MAX(as2.date) FROM asset_sentiment as2 WHERE asset_sentiment.asset_id = as2.asset_id)"
            select_query2 +=" AND user_watchlist.user_id =" + str(user_id)
            select_query2 +=" GROUP BY user_watchlist.asset_id"
            
            cursor.execute(select_query2)
            response2 = cursor.fetchall()
            result["watchlist_sentiment"] = general_utils.sql_to_json(response2, cursor)


        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return result
    
    def get_reports_by_source(self, source_id):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "SELECT * FROM reports WHERE source =" + str(source_id)
            cursor.execute(query)
            result = cursor.fetchall()        
            reports_df = general_utils.sql_to_dataframe(result, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return reports_df  
    
    def get_reports_by_asset_id(self, asset_id):
        try:
            result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            query = "SELECT * FROM reports WHERE asset_id =" + str(asset_id)
            cursor.execute(query)
            result = cursor.fetchall()        
            reports_df = general_utils.sql_to_dataframe(result, cursor)
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return reports_df  

    def get_writer_full_text(self):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT reports.report_id, reports.full_text, reports.writer_reliability_is_processed, reports.writer, writers_reliability.name AS 'writer_name' "
            select_query += "FROM reports "
            select_query += "INNER JOIN writers_reliability ON reports.writer = writers_reliability.writers_reliability_id "
            select_query += "WHERE reports.writer_reliability_is_processed = 0  AND reports.source < 6 AND reports.writer IS NOT NULL"

            cursor.execute(select_query)
            myresult = cursor.fetchall()
            repid = []
            full_text = []
            is_processed = []
            writer_id = []
            writers = []
            for x in myresult:
                rep, f, is_pros, id, n = x
                repid.append(rep)
                full_text.append(f)
                is_processed.append(is_pros)
                writer_id.append(id)
                writers.append(n)
            df = pd.DataFrame(data=list(zip(repid, full_text, is_processed, writer_id, writers)),
                              columns=['report_id','full_text', 'is_processed', 'writer_id','writer'])
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return df

    def get_source_full_text(self):
        try:
            # result = []
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT reports.report_id, reports.full_text, reports.source_reliability_is_processed, reports.source, source_reliability.name AS 'source_name' "
            select_query += "FROM reports "
            select_query += "INNER JOIN source_reliability ON reports.source = source_reliability.source_reliability_id "
            select_query += "WHERE reports.source_reliability_is_processed = 0 AND reports.source < 6 AND reports.source IS NOT NULL"
            cursor.execute(select_query)
            myresult = cursor.fetchall()
            repid = []
            full_text = []
            is_processed = []
            source_id = []
            sources = []
            for x in myresult:
                rep, f, is_pros, id, n = x
                repid.append(rep)
                full_text.append(f)
                is_processed.append(is_pros)
                source_id.append(id)
                sources.append(n)
            df = pd.DataFrame(data=list(zip(repid, full_text, is_processed,source_id, sources)),
                              columns=['report_id','full_text', 'is_processed', 'source_id','source'])
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return df

    def update_writers_reliability(self,  df):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            for i in range(0, len(df)):
                insert_query = """UPDATE writers_reliability SET reliability = {} WHERE writers_reliability_id = {}; """.format(df['score'][i], df['writer_id'][i])
                cursor.execute(insert_query)
                mydb.commit()
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()

    def update_source_reliability(self,  df):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            for i in range(0, len(df)):
                insert_query = """UPDATE source_reliability SET reliability = {} WHERE source_reliability_id = {}; """.format(df['score'][i], df['source_id'][i])
                cursor.execute(insert_query)
                mydb.commit()
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()

    def update_is_processed(self,  df, feature):
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            for i in range(0, len(df)):
                insert_query = """UPDATE reports SET {}_reliability_is_processed = 1 WHERE report_id = {}; """.format(
                   feature, df['report_id'][i])
                cursor.execute(insert_query)
                mydb.commit()
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
    
    def get_source_disclaimer(self, source_name):
        source_reliability_id = -1        
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)
            select_query = "SELECT disclaimer FROM source_reliability WHERE name = '" + str(source_name) + "'"
            cursor.execute(select_query)    
            result = cursor.fetchone()
            disclaimer = result[0]
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return disclaimer

    def testTimestamps(self):
        cmt = general_utils.getCurrentTimestsamp()
        print(cmt)
    

    def get_user_last_value(self, userId):
        resp = {}
        try:
            mydb = self.getDatabaseConnection()
            cursor = mydb.cursor()
            select_query = "SELECT cash, portfolio_assets_daily_value FROM portfolio_history WHERE user_id = " + str(userId) + " ORDER BY date DESC LIMIT 1"
            cursor.execute(select_query) 
            result = cursor.fetchall()
            resp["status"] = "SUCCESS"
            resp["portfolioCurrValue"] = result[0][0] + result[0][1] 
        except Exception as e:
            resp["status"] = "FAILED"
            resp["error"] = "There are indeces with not up-to-dated close values"
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return resp
            # resp["status"] = "SUCCESS"
            # resp["portfolioCurrValue"] = portfolioCurrValue
