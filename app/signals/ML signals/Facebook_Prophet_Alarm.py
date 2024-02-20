import pandas as pd
import numpy as np
from fbprophet import Prophet
import configparser
import os
import pickle
from pathlib import Path
import time

config = configparser.ConfigParser()
config.read('config.ini')
signal_dir = os.path.join(os.getcwd(), "app", "signals", "ML_signals")

class Facebook_Prophet_Emergency():
    def Facebook_Prophet_train(self, df, season, asset):
        df = pd.DataFrame(data=df).reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        df = df[['ds', 'y']]
        if season:
            model = Prophet(daily_seasonality=True)
        else:
            model = Prophet()
        model.fit(df)
        fb_dir = os.path.join(signal_dir, "FB_Prophet_models")
        if not os.path.exists(fb_dir):  # Creates the output directory if it does not exist
            os.makedirs(fb_dir)
        filename = os.path.join(fb_dir, '{}.pkl'.format(asset))
        pickle.dump(model, open(filename, 'wb'))

    def Facebook_Prophet_Forecast(self, asset):
        fb_dir = os.path.join(signal_dir, "FB_Prophet_models")
        filename = os.path.join(fb_dir, '{}.pkl'.format(asset['name']))
        model = pickle.load(open(filename, 'rb'))
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future).set_index('ds')
        return forecast

    def Facebook_Prophet_Emergency_Estimation(self, asset):
        prophet_data = self.Facebook_Prophet_Forecast(asset)[-100:]
        prophet_data['Var_Prophet'] = 0.0
        mean1 = prophet_data['yhat'].mean()
        var_list = []
        for k in range(0, len(prophet_data)):
            var_list.append(pow((prophet_data.iloc[k]['yhat'] - mean1), 2))
        prophet_data['Var_Prophet'] = var_list
        prophet_data['Emer_Prophet'] = 0.0
        var_mean = prophet_data['Var_Prophet'].mean()
        for j in range(0, len(prophet_data)):
            if abs(prophet_data.iloc[j]['Var_Prophet']) > 1.5 * abs(var_mean):
                prophet_data.iloc[j]['Emer_Prophet'] = 1.0
        prophet_data['Return_Prophet'] = 0.0
        return_prophet_list = [0.0]
        for p in range(0, len(prophet_data) - 1):
            return_prophet_list.append(
                (prophet_data.iloc[p + 1]['yhat'] - prophet_data.iloc[p]['yhat']) / prophet_data.iloc[p]['yhat'])
        prophet_data['Return_Prophet'] = return_prophet_list
        std_prophet1 = (prophet_data['Return_Prophet'].std())
        prophet_data['Emerg_std_Prophet'] = 0.0
        prophet_data.index = pd.to_datetime(prophet_data.index)
        Emeg_prophet_list = []
        for j in range(0, len(prophet_data)):
            if (abs(prophet_data.iloc[j]['Return_Prophet']) > 3 * std_prophet1):
                Emeg_prophet_list.append(1.0)
            else:
                Emeg_prophet_list.append(0.0)
        prophet_data['Emerg_std_Prophet'] = Emeg_prophet_list
        return prophet_data[['Emer_Prophet', 'Emerg_std_Prophet']].reset_index(drop=True)



