from ..database.dbManager import DBManager
from ..utilities.generalUtils import GeneralUtils
import time
import pandas as pd
import math
import numpy as np


class PortfolioOverview:
    def __init__(self):
        self.general_utils = GeneralUtils()
        self.db_manager = DBManager()
        self.days_before = 600

    def calcNumOfAssetsToBuy(self, asset_curr_val, budget):
        return math.floor(budget/asset_curr_val * 100) /100

    def calcCostToBuy(self, numOfAssets, asset_curr_val):
        return numOfAssets*asset_curr_val

    def calcSellProfit(self, userId, assetId, close):
        profit = 0
        asset_df = self.db_manager.get_user_asset(userId, assetId)
        amount = asset_df["amount"].sum()
        profit = amount * close
        return profit, amount

    def buyAsset(self, userId, assetId, budget, signal_id=None):
        # 1. check asset current value
        resp = {}
        current_milli_time = self.general_utils.getCurrentTimestsamp()
        close = self.getAssetCurrValue(assetId)         
        if close != None:
            # 2. calculate the num of values that the user can buy
            numOfAssets = self.calcNumOfAssetsToBuy(close, budget)
            if numOfAssets == -1:
                resp["status"] = "FAILED"
                resp["error"] = "Budget is not enough to buy an instance of this asset"
                return resp

            # 3. check if user has the required money
            cost = self.calcCostToBuy(numOfAssets, close)
            user_budget = self.db_manager.get_user_budget(userId)

            if user_budget["cash"] >= cost:
                new_cash_value = user_budget["cash"] - cost
                # TODO: to asset_buy_value na allaksei onoma tha mperdeutoume me to insert_accepted_signal_to_signals_history
                data_to_update = {
                    "assetId": assetId,
                    "date": current_milli_time,
                    "amount": numOfAssets,
                    "asset_buy_value": float(close),
                    "action": 2, #2-->BUY
                    "cash": float(new_cash_value),
                    "signal_id": signal_id
                }

                r1 = self.db_manager.update_db_during_buy_asset(userId, data_to_update)
                if r1["status"] == "FAILED":
                    resp["status"] = "FAILED"
                    resp["error"] = "Error during updating db values at buy asset"
                    return resp

                # 7. Update portfolio percentages
                r2 = self.updateUserAssetsAndPortfolioPercentages(userId)
                if r2["status"] == "FAILED":
                    resp["status"] = "FAILED"
                    resp["error"] = "Error during updating assets percentages"
                    return resp

                
                if signal_id != None:
                    #insert to accepted signals
                    all_amount_value = data_to_update["asset_buy_value"] * data_to_update["amount"]
                    r3 = self.db_manager.insert_accepted_signal_to_signals_history(userId, signal_id, current_milli_time, all_amount_value)
                    if r3["status"] == "FAILED":
                        resp["status"] = "FAILED"
                        resp["error"] = "Error inserting entry into signals acceptance history"
                        return resp

                resp["status"] = "SUCCESS"
                return resp
        else:
            resp["status"] = "FAILED"
            resp["error"] = "Asset Indeces are not up-to-dated"
            return resp

    def sellAsset(self, userId, assetId):
        resp = {}
        current_milli_time = self.general_utils.getCurrentTimestsamp()
        close = self.getAssetCurrValue(assetId)         
        if close != None:

            data_to_update = {}

            user_budget = self.db_manager.get_user_budget(userId)["cash"]
            profit, amount = self.calcSellProfit(userId, assetId, close)
            # 5. Update portfolio cash
            new_cash_value = user_budget + profit


            data_to_update = {
                "assetId": assetId,
                "new_cash_value": new_cash_value,
                "action": 3, # 3-->SELL
                "amount": amount,
                "date": current_milli_time,
                "asset_value": close                
            }

            # 6. pare agores pou sxetizontai me shmata
            asset_df = self.db_manager.get_user_asset(userId, assetId)
            asset_signals_df = asset_df.dropna(subset=['signal_id'])
            if asset_signals_df.empty == False:
                asset_signals_df = asset_signals_df.reset_index()

                asset_signals_df["curr_close"] = close
                asset_signals_df["sell_date"] = current_milli_time
                asset_signals_df["success"] = asset_signals_df.apply(lambda row: True if (row["asset_buy_value"] < row["curr_close"]) else False, axis=1)
                asset_signals_df["buy_value"] = asset_signals_df["asset_buy_value"] * asset_signals_df["amount"]
                asset_signals_df["sell_value"] = asset_signals_df["curr_close"] * asset_signals_df["amount"]

                asset_signals_df["asset_profit_percentage"] = asset_signals_df.apply(lambda row: self.calcProfitPercentage(row["buy_value"], row["sell_value"]), axis=1)
                asset_signals_df["asset_profit"] = asset_signals_df["sell_value"] - asset_signals_df["buy_value"]                        
                
                asset_signals_df["init_port_profit"] = asset_signals_df.apply(lambda row: self.calcPortfolioValueByDate(userId, row["date"])["portfolioValue"], axis=1)
                asset_signals_df["end_port_profit"] = asset_signals_df.apply(lambda row: self.calcPortfolioCurrentValue(userId)["portfolioCurrValue"], axis=1)
                asset_signals_df["portfolio_profit"] = asset_signals_df["end_port_profit"] - asset_signals_df["init_port_profit"]

                asset_signals_df["portfolio_asset_profit_percentage"] = asset_signals_df.apply(lambda row: row["asset_profit"]/row["portfolio_profit"] *100 if row["portfolio_profit"] > 0 else 0, axis=1)

                r3 = self.db_manager.update_signals_acceptance_history(userId, asset_signals_df)
                if r3["status"] == "FAILED":
                    resp["status"] = "FAILED"
                    resp["error"] = "Error during updating update_signals_acceptance_history"
                    return resp


            r1 = self.db_manager.update_db_during_sell_asset(userId, data_to_update)
            if r1["status"] == "FAILED":
                resp["status"] = "FAILED"
                resp["error"] = "Error during updating db values at buy asset"
                return resp

            # 6. Update portfolio percentages
            r2 = self.updateUserAssetsAndPortfolioPercentages(userId)
            if r2["status"] == "FAILED":
                resp["status"] = "FAILED"
                resp["error"] = "Error during updating assets percentages"
                return resp
            else:
                resp = {"cash": new_cash_value, "assets_value": r2["assets_value"]}
                return resp
        else:
            resp["status"] = "FAILED"
            resp["error"] = "Asset Indeces are not up-to-dated"
            return resp

    def calcPortfolioCurrentValue(self, userId):
        resp = {}
        user_budget = self.db_manager.get_user_budget(userId)["cash"]
        user_assets = self.db_manager.get_user_assets(userId)

        user_assets = user_assets.groupby(["asset_id"]).sum().reset_index().drop(columns=["date","user_id","percentage","asset_buy_value"])
        user_assets["curr_value"] = user_assets["asset_id"].apply(self.getAssetCurrValue)
        if user_assets.isnull().values.any():
            resp["status"] = "FAILED"
            resp["error"] = "There are indeces with not up-to-dated close values"
            return resp
        else:
            user_assets["asset_worth"] = user_assets["amount"] * user_assets["curr_value"]
            assets_total_worth = user_assets["asset_worth"].sum()
            portfolioCurrValue = assets_total_worth + user_budget

            resp["status"] = "SUCCESS"
            resp["portfolioCurrValue"] = portfolioCurrValue
            return resp

    def calcPortfolioValueByDate(self, userId, date):
        resp = {}        
        
        user_budget = self.db_manager.get_user_budget_by_date(userId, date)["cash"]
        user_assets = self.db_manager.get_user_assets_history(userId, date)

        user_assets = user_assets.groupby(["asset_id"]).sum().reset_index().drop(columns=["user_id","percentage"])      
        # TODO: na valw elegxo an to from einai apo th mera pou agorasthke h metoxh an nai vazwq thn timh agoras (giati mporei na einai timh close prohgoumenhs hmeras) 
        if user_assets.shape[0] == 0:
            resp["status"] = "SUCCESS"
            resp["portfolioValue"] = user_budget
            return resp
        user_assets["curr_value"] = user_assets.apply(lambda row: self.getAssetValueByDate(row["asset_id"], date), axis=1)
        if user_assets.isnull().values.any():
            resp["status"] = "FAILED"
            resp["error"] = "There are indeces with not up-to-dated close values"
            return resp
        else:
            user_assets["asset_value"] = user_assets.apply(lambda row: row['asset_buy_value'] if row['date'] == date else row['curr_value'], axis=1)
            user_assets["asset_worth"] = user_assets["amount"] * user_assets["asset_value"]
            assets_total_worth = user_assets["asset_worth"].sum()
            portfolioValue = assets_total_worth + user_budget
            
            resp["status"] = "SUCCESS"
            resp["portfolioValue"] = portfolioValue
            return resp

    def calcProfitPercentage(self, initialValue, currentValue):
        if initialValue > 0:
            return ((currentValue-initialValue)/initialValue)*100
        else:
            return None
    
    def calcPortfolioProfitPercentage(self, initialValue, currentValue):
        if initialValue > 0:
            return ((currentValue-initialValue)/initialValue)*100
        else:
            return None
    
    def calcPortfolioProfit(self, initialValue, currentValue):
        if initialValue > 0: 
            return currentValue-initialValue
        else:
            return None

    def getAssetCurrValue(self, assetId):
        current_milli_time = self.general_utils.getCurrentTimestsamp()
        from_msecs = current_milli_time - self.general_utils.convertDaysToMiliseconds(self.days_before)    
        asset_curr_value = self.db_manager.get_asset_indices(assetId, from_msecs, current_milli_time) 
        if "asset_indices" in asset_curr_value and len(asset_curr_value["asset_indices"]) > 0:
            #pairnw shmerinh kai 30-meres prin timh se periptwsh pou den exw shmerinh na douleve me palia
            # an den exei kanena apo ta 2 lew oti den einai updated oi times tis metoxhs
            df = pd.DataFrame(asset_curr_value["asset_indices"])
            close = float(df.nlargest(1, ['date'])["close"])
            return close
        else:
            return None

    def getAssetValueByDate(self, assetId, date):
        # pare oles tis times apo shmera mexri 1 mhna prin kai krata thn teleutaia
        from_msecs = date - self.general_utils.convertDaysToMiliseconds(self.days_before)
        asset_curr_value = self.db_manager.get_asset_indices(assetId, from_msecs, date)
        if "asset_indices" in asset_curr_value and len(asset_curr_value["asset_indices"]) > 0:
            df = pd.DataFrame(asset_curr_value["asset_indices"])
            close = df.sort_values(by=["date"], ascending=False).iloc[0]["close"]
            return close
        else:
            return None

    def getPortfolioOverview(self, userId, from_date, to_date):
        portfolio_register_date = self.db_manager.get_portfolio_register_date(userId)
        
        #TODO: EDW MPIKE MONO GIA TO SCREENSHOT 
        curr_val = self.calcPortfolioCurrentValue(userId) #self.db_manager.get_user_last_value(userId)
        min_max_gain = self.getPortfoliosAssetWithMinMaxGain(userId, from_date, to_date)

        resp = {}
        
        resp["min_value"] = self.getPortfolioMinValue(userId, from_date, to_date)
        resp["max_value"] = self.getPortfolioMaxValue(userId, from_date, to_date)  
        resp["min_return_asset"] = min_max_gain["min_gain_asset_name"]
        resp["max_return_asset"] = min_max_gain["max_gain_asset_name"]
        resp["min_return"] = min_max_gain["min_gain"]
        resp["max_return"] = min_max_gain["max_gain"]
        resp["sharpe_ratio"] = self.getSharpeRatio(userId, from_date, to_date)
        resp["max_loss_percentage"] = self.getGreaterLoss(userId, from_date, to_date)


        if from_date == portfolio_register_date:
            resp["inital_value"] = 100000 # Arxikh aksia xartofulakiou kata thn eggrafh einai 100000 gia olous
        else:        
            portfolioValue_resp = self.calcPortfolioValueByDate(userId, from_date) 
            resp["inital_value"] = portfolioValue_resp["portfolioValue"] if portfolioValue_resp["status"] == "SUCCESS" else None
        resp["current_value"] = curr_val["portfolioCurrValue"] if curr_val["status"] == "SUCCESS" else None
        resp["total"] = self.calcPortfolioProfit(resp["inital_value"], resp["current_value"]) if resp["current_value"] != None else None
        resp["total_percentage"] = self.calcPortfolioProfitPercentage(resp["inital_value"], resp["current_value"]) if resp["current_value"] != None else None
        return resp

    def getPortfolioMaxValue(self, userId, from_date, to_date):
        portfolio_register_date = self.db_manager.get_portfolio_register_date(userId)
        portfolio_history = self.db_manager.get_portfolio_history(userId, from_date, to_date)
        if portfolio_register_date == from_date:
            new_row = {'portfolio_history_id': 0, 
                       'user_id': userId, 
                       'date': portfolio_register_date, 
                       'profile_type':1, 
                       'optimization_method':1, 
                       'cash':100000,
                       'investment_percentage':0, 
                       'invested_money':0,
                       'portfolio_assets_daily_value':0}
            portfolio_history = portfolio_history.append(new_row, ignore_index=True)
        portfolio_history["portfolio_total_value"] = portfolio_history["portfolio_assets_daily_value"] + portfolio_history["cash"]
        max_value = portfolio_history["portfolio_total_value"].max()
        return max_value

    def getPortfolioMinValue(self, userId, from_date, to_date):
        portfolio_register_date = self.db_manager.get_portfolio_register_date(userId)
        portfolio_history = self.db_manager.get_portfolio_history(userId, from_date, to_date)

        # an to from einai h mera dhmiourgias prosthese kai akoma mia grammh me to arxiko poso tou xartofulakio (dhl 100000)
        if portfolio_register_date == from_date:
            new_row = {'portfolio_history_id': 0, 
                       'user_id': userId, 
                       'date': portfolio_register_date, 
                       'profile_type':1, 
                       'optimization_method':1, 
                       'cash':100000,
                       'investment_percentage':0, 
                       'invested_money':0,
                       'portfolio_assets_daily_value':0}
            portfolio_history = portfolio_history.append(new_row, ignore_index=True)

        portfolio_history["portfolio_total_value"] = portfolio_history["portfolio_assets_daily_value"] + portfolio_history["cash"]
        min_value = portfolio_history["portfolio_total_value"].min()
        return min_value

    def getPortfoliosAssetWithMinMaxGain(self, userId, from_date, to_date):
        try:
            resp = {}
            history = self.db_manager.get_user_history(userId, from_date, to_date)            

            user_assets_from_date = self.db_manager.get_user_assets_history(userId, from_date)
            user_assets_previous_from_date = self.db_manager.get_user_assets_history(userId, from_date - self.general_utils.convertDaysToMiliseconds(1))
            user_assets_to_date = self.db_manager.get_user_assets(userId)

            # 1. Vres metoxes pou eixame pare dwse auto to mhna
            # einai oti egine mesa sto history (agora, pwlhsh) + tis metoxes pou eixame apo prin sto portfolio
            assets = history["asset_id"].unique()
            if user_assets_previous_from_date["asset_id"].unique().size > 0:
                #BUG otan den mpainei edw mesa einai float ama mpei ginete integer kai ta pinei
                assets = np.unique(np.append(assets, user_assets_previous_from_date["asset_id"].unique()))
                
            assets = assets.astype("float64")
            #BUG TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe'' 
            assets = assets[np.logical_not(np.isnan(assets))]

            assets_information = []

            for asset_id in assets:
            # 2. Gia kathe meetoxh
                #pare history gia to sugekrimeno asset
                asset_history = history[history["asset_id"] == asset_id]
                # vres an htan agorasmeno apo prin
                init_value = 0
                total_investment = 0
                total_sells_gain = 0
                curr_value = 0
                asset_gain = 0
                close = 0
                previous_init_value = 0

                if asset_id in user_assets_previous_from_date["asset_id"].values and asset_id in user_assets_from_date["asset_id"].values:
                    u_asset_previous_from_date = user_assets_previous_from_date[user_assets_previous_from_date["asset_id"] == asset_id]
                    from_date = from_date - self.general_utils.convertDaysToMiliseconds(2)
                    from_date_init = from_date - self.general_utils.convertDaysToMiliseconds(self.days_before - 1)
                    asset_curr_value = self.db_manager.get_asset_indices(asset_id, from_date_init, from_date)
                    if "asset_indices" in asset_curr_value and len(asset_curr_value["asset_indices"]) > 0:
                        df = pd.DataFrame(asset_curr_value["asset_indices"])
                        close = df.sort_values(by=["date"], ascending=False).iloc[0]["close"]
                    else:
                        print("Error: at getPortfoliosAssetWithMinMaxGain: No available asset indices for that period")
                        resp["satus"] = "FAILED"
                        return resp
                    previous_init_value = u_asset_previous_from_date["amount"].sum() * close


                # if asset_id in user_assets_from_date["asset_id"].values:
                #     #pare apo ekeinh thn hmeromhnia mono to sugekrimeno asset kai vgale aksia metoxis
                #     # gia to xartofulakio
                #     u_asset_from_date = user_assets_from_date[user_assets_from_date["asset_id"] == asset_id]
                #     # indices = self.db_manager.get_asset_indices(asset_id, from_date, from_date)
                #     #to kanw mia mera pisw gia na min exw thema init_value kai sto total_investment
                #     from_date = from_date - self.general_utils.convertDaysToMiliseconds(1)
                #     from_date_init = from_date - self.general_utils.convertDaysToMiliseconds(30)
                #     asset_curr_value = self.db_manager.get_asset_indices(asset_id, from_date_init, from_date)
                #     if "asset_indices" in asset_curr_value and len(asset_curr_value["asset_indices"]) > 0:
                #         df = pd.DataFrame(asset_curr_value["asset_indices"])
                #         close = df.sort_values(by=["date"], ascending=False).iloc[0]["close"]
                #     else:
                #         resp["satus"] = "FAILED"
                #         return resp
                #     # -Vres arxikh aksia px 10metox * from_date value (1$) = 10 (opern metoxhs)
                #     init_value = u_asset_from_date["amount"].sum() * close

                # -Vres ependysh gia sugekrimenh metoxh enros diasthmatos
                asset_investment = asset_history[asset_history["action"] == 2]
                asset_investment["investment_value"] = asset_investment["amount"] * asset_investment["asset_value"]
                total_investment = asset_investment["investment_value"].sum()

                # -An exoume sells prosthese poso pou sugentrethhke
                sells = asset_history[asset_history["action"] == 3]
                sells["sell_gain"] = sells["amount"] * sells["asset_value"]
                total_sells_gain = sells["sell_gain"].sum()

                close_to = 0
                if asset_id in user_assets_to_date["asset_id"].values:
                    u_asset_to_date = user_assets_to_date[user_assets_to_date["asset_id"] == asset_id]
                    to_date_init = to_date - self.general_utils.convertDaysToMiliseconds(self.days_before)
                    asset_curr_value = self.db_manager.get_asset_indices(asset_id, to_date_init, to_date)
                    if "asset_indices" in asset_curr_value and len(asset_curr_value["asset_indices"]) > 0:
                        df = pd.DataFrame(asset_curr_value["asset_indices"])
                        close_to = df.sort_values(by=["date"], ascending=False).iloc[0]["close"]
                    else:
                        print("Error: at getPortfoliosAssetWithMinMaxGain: No available asset indices for that period")
                        resp["satus"] = "FAILED"
                        return resp
                    curr_value = u_asset_to_date["amount"].sum() * close_to

                # - Kerdos = Curr Aksia + sell_gains - ependush - arxik aksia
                # TODO: tha prepei na proseksoume ti ginetai me tis metoxes pou agorasthkan thn idia mera.
                # gia to timestamp. Den tha prepei na upologizontai kai to init_value kai sto total_investment
                # auto einai bug twra opws to xw
                # na afairw ena milisec kai tha einai boba
                # epishs prepei na doume ti tha ginei me open, close kai ti otan den tha exoume times sth vash --> OK
                asset_gain = curr_value + total_sells_gain - total_investment - previous_init_value

                asset_info = {}
                asset_info["asset_id"] = asset_id
                asset_info["asset_gain"] = asset_gain
                assets_information.append(asset_info)

            # vres max kai min
            user_assets_df = pd.DataFrame(assets_information)

            user_assets_df.iloc[user_assets_df["asset_gain"].idxmax()]
            user_assets_df.iloc[user_assets_df["asset_gain"].idxmin()]

            resp["min_gain"] = user_assets_df.iloc[user_assets_df["asset_gain"].idxmin()]["asset_gain"]
            resp["min_gain_asset_name"] = self.db_manager.get_asset_by_id(user_assets_df.iloc[user_assets_df["asset_gain"].idxmin()]["asset_id"])["name"]
            resp["max_gain"] = user_assets_df.iloc[user_assets_df["asset_gain"].idxmax()]["asset_gain"]
            resp["max_gain_asset_name"] = self.db_manager.get_asset_by_id(user_assets_df.iloc[user_assets_df["asset_gain"].idxmax()]["asset_id"])["name"]
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return resp
    
    def getSharpeRatio_old(self, userId, from_date, to_date):
        try:
            risk_free_rate = 0 #Initialized by us
            sharpeRatio = None
            portfolio_history = self.db_manager.get_portfolio_history(userId, from_date, to_date)
            portfolio_history["total_value"] = portfolio_history["portfolio_assets_daily_value"] + portfolio_history["cash"]
            std = portfolio_history["total_value"].std()
            mean = portfolio_history["total_value"].mean()
            std_annualized = std * 252
            mean_annualized = mean * 252
            if ((not math.isnan(std_annualized)) and (std_annualized !=0)):
                sharpeRatio = (mean_annualized - risk_free_rate)/std_annualized
            else:            
                sharpeRatio = None
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return sharpeRatio

    def getSharpeRatio(self, userId, from_date, to_date):
        try:
            risk_free_rate = 0 #Initialized by us
            sharpeRatio = None
            portfolio_history = self.db_manager.get_portfolio_history(userId, from_date, to_date)
            portfolio_register_date = self.db_manager.get_portfolio_register_date(userId)
            # an to from einai h mera dhmiourgias prosthese kai akoma mia grammh me to arxiko poso tou xartofulakio (dhl 100000)
            if portfolio_register_date == from_date:
                data = []
                new_row = {'portfolio_history_id': 0, 
                        'user_id': userId, 
                        'date': portfolio_register_date, 
                        'profile_type':1, 
                        'optimization_method':1, 
                        'cash':100000,
                        'investment_percentage':0, 
                        'invested_money':0,
                        'portfolio_assets_daily_value':0}
                data.insert(0, new_row)
                portfolio_history = pd.concat([pd.DataFrame(data), portfolio_history], ignore_index=True)

            portfolio_history["total_value"] = portfolio_history["portfolio_assets_daily_value"] + portfolio_history["cash"]

            if len(portfolio_history["total_value"].values) > 0:
                initial_portfolio_value = portfolio_history["total_value"].values[0]
                portfolio_history["diff_perc"] = ((portfolio_history["total_value"] - initial_portfolio_value)/(initial_portfolio_value))*100

                std = portfolio_history["diff_perc"].std()
                mean = portfolio_history["diff_perc"].mean()
                std_annualized = std * 252
                mean_annualized = mean * 252
                if ((not math.isnan(std_annualized)) and (std_annualized !=0)):
                    sharpeRatio = (mean_annualized - risk_free_rate)/std_annualized
                else:            
                    sharpeRatio = None
            else:
                return 0

        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return sharpeRatio
    
    def getGreaterLoss(self, userId, from_date, to_date):
        try:
            portfolio_history = self.db_manager.get_portfolio_history(userId, from_date, to_date)

            portfolio_register_date = self.db_manager.get_portfolio_register_date(userId)
            # an to from einai h mera dhmiourgias prosthese kai akoma mia grammh me to arxiko poso tou xartofulakio (dhl 100000)
            if portfolio_register_date == from_date:
                data = []
                new_row = {'portfolio_history_id': 0, 
                        'user_id': userId, 
                        'date': portfolio_register_date, 
                        'profile_type':1, 
                        'optimization_method':1, 
                        'cash':100000,
                        'investment_percentage':0, 
                        'invested_money':0,
                        'portfolio_assets_daily_value':0}
                data.insert(0, new_row)
                portfolio_history = pd.concat([pd.DataFrame(data), portfolio_history], ignore_index=True)

            portfolio_history["total_value"] = portfolio_history["portfolio_assets_daily_value"] + portfolio_history["cash"]
            portfolio_history['total_value_previous'] = portfolio_history['total_value'].shift(1)
            portfolio_history = portfolio_history.iloc[1:]
            portfolio_history["diff"] = portfolio_history["total_value"] - portfolio_history["total_value_previous"]
            portfolio_history["diff_perc"] = (portfolio_history["diff"]/portfolio_history["total_value_previous"])*100
            greaterLoss = portfolio_history["diff_perc"].min()
            if greaterLoss > 0.0: 
                greaterLoss = 0.0
            if math.isnan(greaterLoss):
                greaterLoss = None
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return greaterLoss

    def updateUserAssetsAndPortfolioPercentages(self, userId):
        resp = {}
        user_budget = self.db_manager.get_user_budget(userId)["cash"]
        user_assets = self.db_manager.get_user_assets(userId)
        user_portfolio = self.db_manager.get_user_portfolio(userId)
        if user_assets.empty == False:
            #user_assets = user_assets.groupby(["asset_id"]).sum().reset_index().drop(columns=["date","user_id","asset_buy_value"])
            user_assets["curr_value"] = user_assets["asset_id"].apply(self.getAssetCurrValue)
            if user_assets["curr_value"].isnull().values.any():
                resp["status"] = "FAILED"
                resp["error"] = "There are indeces with not up-to-dated close values"
                return resp
            else:
                user_assets["asset_worth"] = user_assets["amount"] * user_assets["curr_value"]
                assets_total_worth = user_assets["asset_worth"].sum()
                portfolioCurrValue = assets_total_worth + user_budget
                user_assets["percentage"] = user_assets.apply(lambda row: row["asset_worth"]/portfolioCurrValue *100, axis=1)

                #portfolio_percentages
                portfolioCurrValue = assets_total_worth + user_budget
                cash_percentage = user_budget/portfolioCurrValue * 100
                investment_percentage = assets_total_worth/portfolioCurrValue * 100

                r1 = self.db_manager.update_user_assets_percentages(userId, user_assets)
                if r1["status"] == "FAILED":
                    resp["status"] = "FAILED"
                    resp["error"] = "Error while updating user_assets percentages"
                    return resp

                user_portfolio = user_portfolio.iloc[0]

                r2 = self.db_manager.update_user_portfolio(userId, user_portfolio["profile_type"], user_portfolio["optimization_method"], user_portfolio["cash"],investment_percentage, user_portfolio["invested_money"], assets_total_worth)
                if r2["status"] == "FAILED":
                    resp["status"] = "FAILED"
                    resp["error"] = "Error while updating portfolio percentages"
                    return resp
                
                r3 = self.db_manager.update_portfolio_history(userId)
                if r3["status"] == "FAILED":
                    resp["status"] = "FAILED"
                    resp["error"] = "Error while updating portfolio_history percentages"
                    return resp

                resp["status"] = "SUCCESS"
                resp["assets_value"] = assets_total_worth
        else:
                assets_total_worth = 0
                portfolioCurrValue = assets_total_worth + user_budget
                cash_percentage = user_budget/portfolioCurrValue * 100
                investment_percentage = assets_total_worth/portfolioCurrValue * 100

                user_portfolio = user_portfolio.iloc[0]

                r2 = self.db_manager.update_user_portfolio(userId, user_portfolio["profile_type"], user_portfolio["optimization_method"], user_portfolio["cash"], investment_percentage, user_portfolio["invested_money"], assets_total_worth)
                if r2["status"] == "FAILED":
                    resp["status"] = "FAILED"
                    resp["error"] = "Error while updating portfolio percentages"
                    return resp

                r3 = self.db_manager.update_portfolio_history(userId)
                if r3["status"] == "FAILED":
                    resp["status"] = "FAILED"
                    resp["error"] = "Error while updating portfolio_history percentages"
                    return resp

                resp["status"] = "SUCCESS"
                resp["assets_value"] = assets_total_worth
        return resp
    
    def getHistoricalPortfolioProfit(self, userId, from_date, to_date):
        try:
            #TODO: Yparxei bug edw
            portfolio_register_date = self.db_manager.get_portfolio_register_date(userId)
            resp = {}
            portfolio_history = self.db_manager.get_portfolio_history(userId, from_date, to_date)
            if portfolio_register_date == from_date:
                data = []
                new_row = {'portfolio_history_id': 0, 
                        'user_id': userId, 
                        'date': portfolio_register_date, 
                        'profile_type':1, 
                        'optimization_method':1, 
                        'cash':100000,
                        'investment_percentage':0, 
                        'invested_money':0,
                        'portfolio_assets_daily_value':0}
                data.insert(0, new_row)
                portfolio_history = pd.concat([pd.DataFrame(data), portfolio_history], ignore_index=True)

            portfolio_history["total_value"] = portfolio_history["portfolio_assets_daily_value"] + portfolio_history["cash"]
            init_value = portfolio_history.iloc[0]["total_value"]
            # portfolio_history = portfolio_history.iloc[1:]
            portfolio_history["diff"] = portfolio_history["total_value"] - init_value
            portfolio_history = portfolio_history.sort_values(by=['date'])
            
            resp["label"] = portfolio_history.apply(lambda row: self.general_utils.convertMillisecsToDate(row["date"]), axis=1).to_list()
            resp["data"] = portfolio_history["diff"].to_list()
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return resp

    def updateAssetsAndPortfolioPercentagesForAllUser(self):
        success = False
        try:
            all_users = self.db_manager.get_all_users()
            for user in all_users:
                self.updateUserAssetsAndPortfolioPercentages(user["user_id"])
                print(str(user["user_id"]) + " | Percentages Updated")
            success = True
            print("-------- Update Assets And Portfolio Percentages --------")
        except Exception as e:
            print("====================" + str(e) + "====================")
            print("-------- PROBLEM: Update Assets And Portfolio Percentages NOT Completed--------")
            success = False
        finally:
            return success

    def getAllAssetsWithCurrentValues(self):
        response = []
        try:
            all_assets = self.db_manager.get_all_assets()
            for asset in all_assets:
                value = self.getAssetCurrValue(asset["asset_id"])
                if value != None:
                    response.append({"asset": asset, "value": value})
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            return response