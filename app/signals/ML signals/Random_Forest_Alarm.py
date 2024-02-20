import time
from datetime import datetime, timedelta
import configparser
import os
import pandas as pd
from app.signals.ML_signals.utils import pandas_techinal_indicator as ta #https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
from sklearn.ensemble import RandomForestRegressor
import pickle

config = configparser.ConfigParser()
config.read('config.ini')
signal_dir = os.path.join(os.getcwd(), "app", "signals", "ML_signals")

class random_forest_alarm():

    def feature_extraction(self, data):
        for x in [5, 14, 26, 44, 66]:
            data = ta.relative_strength_index(data, n=x)
            data = ta.stochastic_oscillator_d(data, n=x)
            data = ta.williams(data, n=x)
            data = ta.accumulation_distribution(data, n=x)
            data = ta.average_true_range(data, n=x)
            data = ta.momentum(data, n=x)
            data = ta.money_flow_index(data, n=x)
            data = ta.rate_of_change(data, n=x)
            data = ta.on_balance_volume(data, n=x)
            data = ta.commodity_channel_index(data, n=x)
            data = ta.ease_of_movement(data, n=x)
            data = ta.trix(data, n=x)
            data = ta.vortex_indicator(data, n=x)
        data['ema50'] = data['Close'] / data['Close'].ewm(50).mean()
        data['ema21'] = data['Close'] / data['Close'].ewm(21).mean()
        data['ema14'] = data['Close'] / data['Close'].ewm(14).mean()
        data['ema5'] = data['Close'] / data['Close'].ewm(5).mean()

        # Williams %R is missing
        data = ta.macd(data, n_fast=12, n_slow=26)
        return data


    def compute_prediction_int(self, df, n):
        df['pred'] = 0
        for i in range(0, len(df) - 1):
            if df.iloc[i + 1]['Close'] > df.iloc[i]['Close']:
                df.loc[i, 'pred'] = 1
        return pred.astype(int)  # df['pred'].values#pred.astype(int


    def prepare_data(self, df, horizon):
        data = self.feature_extraction(df).dropna().iloc[:-horizon]
        return data.dropna()


    def get_exp_preprocessing(self, df, alpha=0.1):
        edata = df.ewm(alpha=alpha).mean()
        return edata


    def random_forest_train(self, data, asset):
        data = data.reset_index()
        data['pred'] = data['Close'].diff()
        data['pred'] = data[data['pred'] > 0]['pred'].apply(lambda x: 1)
        data['pred'] = data['pred'].fillna(0)
        data = self.feature_extraction(data).iloc[:-1].fillna(0)
        cols = data.columns.values.tolist()
        a, b = cols.index('pred'), cols.index('MACDdiff_12_26')
        cols[b], cols[a] = cols[a], cols[b]
        data = data[cols]
        del (data['Open'])
        del (data['High'])
        del (data['Low'])
        del (data['Volume'])
        first_data = self.get_exp_preprocessing(data.iloc[:, :-1])
        pred_data = data.iloc[:, -1]
        data = [first_data, pred_data]
        data = pd.concat(data, axis=1)
        y = data['Close']
        # remove the output from the input
        features = [x for x in data.columns if x not in ['gain', 'Close', 'Date']]
        X = data[features].fillna(0)
        rf = RandomForestRegressor(n_jobs=-1, n_estimators=65, random_state=42)
        model = rf.fit(X, y)
        rf_dir = os.path.join(signal_dir, "RF_models")
        if not os.path.exists(rf_dir):  # Creates the output directory if it does not exist
            os.makedirs(rf_dir)
        filename = os.path.join(rf_dir, '{}.pkl'.format(asset['name']))
        pickle.dump(model, open(filename, 'wb'))

    def random_forest_inference(self, data, asset):
        data = data.reset_index()
        date = data['Date'].copy()
        data['pred'] = data['Close'].diff()
        data['pred'] = data[data['pred'] > 0]['pred'].apply(lambda x: 1)
        data['pred'] = data['pred'].fillna(0)
        data = self.feature_extraction(data).iloc[:-1].fillna(0)
        cols = data.columns.values.tolist()
        a, b = cols.index('pred'), cols.index('MACDdiff_12_26')
        cols[b], cols[a] = cols[a], cols[b]
        data = data[cols]
        del (data['Open'])
        del (data['High'])
        del (data['Low'])
        del (data['Volume'])
        first_data = self.get_exp_preprocessing(data.iloc[:, :-1])
        pred_data = data.iloc[:, -1]
        data = [first_data, pred_data]
        data = pd.concat(data, axis=1)
        y = data['Close']
        # remove the output from the input
        features = [x for x in data.columns if x not in ['gain', 'Close', 'Date']]
        X = data[features].fillna(0)
        rf_dir = os.path.join(signal_dir, "RF_models")
        filename = os.path.join(rf_dir, '{}.pkl'.format(asset['name']))
        model = pickle.load(open(filename, 'rb'))
        pred = model.predict(X)
        predict = pd.DataFrame({'pred_close_RF': y}).reset_index(drop=True)
        my_frame = [date.reset_index(drop=True).to_frame().iloc[-len(pred) + 1:].reset_index(drop=True), predict]
        mydata = pd.concat(my_frame, axis=1)
        mydata['Date'].iloc[-1] = (datetime.today())
        mydata = mydata.append({'Date': datetime.today() + timedelta(days=1), 'pred_close_RF':pred[len(pred)-1]}, ignore_index=True)
        mydata = mydata.set_index(['Date'])
        mean = mydata['pred_close_RF'].mean()
        mydata['Var_RF'] = 0.0
        var_list = []
        for m in range(0, len(mydata)):
            var_list.append(pow((mydata.iloc[m]['pred_close_RF'] - mean), 2))
        mydata['Var_RF'] = var_list
        var_mean = mydata['Var_RF'].mean()
        mydata['Emer_RF'] = 0.0
        Emer_RF_list_Var = []
        for k in range(0, len(mydata)):
            if abs(mydata.iloc[k]['Var_RF']) > 1.5 * abs(var_mean):
                Emer_RF_list_Var.append(1.0)
            else:
                Emer_RF_list_Var.append(0.0)
        mydata['Emer_RF'] = Emer_RF_list_Var
        mydata['Return_RF'] = 0.0
        return_RF_list = [0.0]
        for p in range(0, len(mydata) - 1):
            return_RF_list.append(
                (mydata.iloc[p + 1]['pred_close_RF'] - mydata.iloc[p]['pred_close_RF']) / mydata.iloc[p]['pred_close_RF'])
        mydata['Return_RF'] = return_RF_list
        std_RF = mydata['Return_RF'].std()
        mydata['Emerg_std_RF'] = 0.0
        mydata.index = pd.to_datetime(mydata.index)
        Emeg_RF_list = []
        for j in range(0, len(mydata)):
            if (abs(mydata.iloc[j]['Return_RF']) > 3 * std_RF):
                Emeg_RF_list.append(1.0)
            else:
                Emeg_RF_list.append(0.0)
        mydata['Emerg_std_RF'] = Emeg_RF_list
        return mydata[['Emer_RF','Emerg_std_RF']].reset_index(drop = True)

