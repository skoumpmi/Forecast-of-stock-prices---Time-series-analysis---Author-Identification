import mysql.connector as mySQL
import configparser
from pandas_datareader import data as pdr
import pandas as pd
from ..database.dbManager import DBManager
from ..utilities.generalUtils import GeneralUtils

config = configparser.ConfigParser()
config.read('config.ini')

class AssetIndicesStreamer:
    def __init__(self):
        self.db_manager = DBManager()
        self.utils = GeneralUtils()

    def download_stock_indices(self, asset, start_day, end_day):  
        try:      
            df = pd.DataFrame()
            df = pdr.get_data_yahoo(asset["yahoo_ticker"], start=start_day, end=end_day).reset_index()
        except Exception as e:
            # print("====================" + str(e) + "====================")
            print(start_day + " | " + asset["name"] + " | Asset Indices NOT ready yet") 
        finally:
            return df

    def download_stock_indices_for_all_assets(self):
        success = False
        try:
            current_date = self.utils.getCurrentDate()
            start_day = current_date
            end_day = current_date
            rep_manager = DBManager()
            allAssets = rep_manager.get_all_assets()
            for asset in allAssets:           
                data = self.download_stock_indices(asset, start_day, end_day)
                if not data.empty:
                    for index, row in data.iterrows():
                        self.db_manager.insert_asset_indices(asset, row, index)
            success = True
            print("-------- Update Assets Indices --------")
        except Exception as e:
            print("====================" + str(e) + "====================")
            print("-------- PROBLEM: Update Assets Indices NOT Completed--------")
            success = False
        finally:
            return success
