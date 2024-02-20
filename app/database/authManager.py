import mysql.connector as mySQL
import configparser
import hashlib
import pandas as pd

from ..portfolio.portfolioOverview import PortfolioOverview
from ..utilities.generalUtils import GeneralUtils
from ..database.dbManager import DBManager

config = configparser.ConfigParser()
config.read('config.ini')


class AuthManager:

    def __init__(self):
        self.po = PortfolioOverview()
        self.general_utils = GeneralUtils()
        self.db_manager= DBManager()
    

    def store_user(self, data):
        try:
            mydb = self.db_manager.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)

            select_query = "SELECT email FROM user WHERE email = \'" + data["email"] + "'"
            cursor.execute(select_query)
            select_result=cursor.fetchone()
            if select_result == None:
                insert_query = "INSERT INTO user (email, pass, roles) VALUES(%s, %s, %s)"
                hashed_pass = hashlib.sha256(data["password"].encode('utf-8')).hexdigest()
                insert_args = (data["email"], hashed_pass, "user")
                cursor.execute(insert_query, insert_args)
                mydb.commit()
            user = {"email": data["email"], "role": ["user"]}
        except Exception as e:
            print("====================" + str(e) + "====================")
            user = {}
        finally:
            cursor.close()
            mydb.close()
            return user

    def retrieve_user(self, data):
        
        user = {}
        message = ""
        try:
            mydb = self.db_manager.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)

            select_query = "SELECT * FROM user WHERE email = \'" + data["email"] + "'"
            cursor.execute(select_query)
            select_result=cursor.fetchone()
            if select_result != None:
                if data["password"] == select_result[2]:
                    user = {"email": select_result[1], "role": select_result[3].split()}
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return user
    
    def retrieve_user_by_email(self, email):
        '''
            func that returns user
            input: user email
        '''
        
        user = {}
        message = ""
        try:
            mydb = self.db_manager.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)
            select_query = "SELECT * FROM user WHERE email = \'" + email + "'"
            cursor.execute(select_query)
            select_result = cursor.fetchone()
            if select_result != None:                
                user = {"user_id": select_result[0], "email": select_result[1], "role": select_result[3].split()}
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return user

    def save_user_portfolio(self, data):
        user = {}
        message = ""
        try:
            mydb = self.db_manager.getDatabaseConnection()
            cursor = mydb.cursor(buffered=True)
            cursor.execute("SELECT user_id FROM user WHERE email = \'" + data["user_email"] + "'")
            tmp_result = cursor.fetchone()
            if tmp_result != None:  
                user_id = tmp_result[0]

                cursor.execute("SELECT * FROM portfolio WHERE user_id = \'" + str(user_id) + "'")
                portfolio = cursor.fetchall()
                if len(portfolio) == 0:
                    date = self.general_utils.getCurrentTimestsamp()
                    date_start, date_end = self.general_utils.getStartEndFromDate(date)
                
                    #Insert Data to Portfolio Table
                    portfolio_data = (user_id, data["risk_profile"]["risk_profile_id"], data["optimization_method"]["opt_method_id"], data["invested_money"], data["cash"])
                    cursor.execute("INSERT INTO portfolio (user_id, profile_type, optimization_method, invested_money, cash) VALUES (%s, %s, %s, %s, %s)", portfolio_data)
                
                    portfolio_history_data = (user_id, date_start, data["risk_profile"]["risk_profile_id"], data["optimization_method"]["opt_method_id"], data["invested_money"], data["cash"])
                    cursor.execute("INSERT INTO portfolio_history (user_id, date, profile_type, optimization_method, invested_money, cash) VALUES (%s, %s, %s, %s, %s, %s)", portfolio_history_data)

                    action = 1             
                    history_data = (user_id, action, date, data["invested_money"])
                    cursor.execute("INSERT INTO history (user_id, action, date, amount) VALUES (%s, %s, %s, %s)", history_data)
                    mydb.commit()
                
                    for portfolio_asset in data["portfolio_assets"]:
                        self.po.buyAsset(user_id, portfolio_asset["asset_id"], portfolio_asset["asset_pieces_value"])
                
                    #Insert Data to user_watchlist Table
                    watchlist_assets_data = []
                    for watchlist_asset in data["watchlist_assets"]:
                        watchlist_assets_data.append((user_id, watchlist_asset["asset_id"]))
                    cursor.executemany("INSERT INTO user_watchlist (user_id, asset_id) VALUES (%s, %s)", watchlist_assets_data)

                    #Update role of user after correct portfolio registration
                    role = "investor"
                    cursor.execute("UPDATE user SET roles = \'" + role + "' WHERE user_id = \'" + str(user_id) + "'")
    
                    mydb.commit()

        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
            return user


    def edit_user_portfolio(self, data, user_id):
        success = False
        try:
          
            today_timestamp = self.general_utils.getCurrentTimestsamp()
            date_start, date_end = self.general_utils.getStartEndFromDate(today_timestamp)

            portfolio = self.db_manager.get_user_portfolio(user_id)


            mydb = self.db_manager.getDatabaseConnection()
            cursor = mydb.cursor()
            
            sql_query = "UPDATE portfolio SET profile_type = " + str(data["risk_profile_id"]) + ", optimization_method = " + str(data["opt_method_id"]) +" WHERE user_id = " + str(user_id);
            cursor.execute(sql_query)

            cursor.execute("SELECT * FROM portfolio_history WHERE user_id = \'" + str(user_id) + "'" + " AND date >=" + str(date_start) + " AND date <=" + str(date_end))
            portfolio_history = cursor.fetchall()
            row = portfolio.iloc[0]
            if len(portfolio_history) == 0:
                # an den uparxei grapse kainourgio entry
                sql = "INSERT INTO portfolio_history (user_id, date, profile_type, optimization_method, cash, investment_percentage, invested_money, portfolio_assets_daily_value)" \
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                values = (user_id, date_start, int(data["risk_profile_id"]), int(data["opt_method_id"]), float(row["cash"]), float(row["investment_percentage"]), float(row["invested_money"]), float(row["portfolio_assets_value"]))
                cursor.execute(sql, values)
            else:
                sql6 =  " UPDATE portfolio_history SET profile_type = " + str(data["risk_profile_id"]) + ", optimization_method = " + str(data["opt_method_id"]) +" WHERE user_id = " + str(user_id) + " AND date >=" + str(date_start) + " AND date <=" + str(date_end) 
                cursor.execute(sql6)
            mydb.commit()

            for asset in data["new_assets_to_buy"]:
                    self.po.buyAsset(user_id, asset["asset_id"], asset["budget"])

            success = True
        except Exception as e:
            print("====================" + str(e) + "====================")
            success = False
        finally:
            cursor.close()
            mydb.close()
            return success