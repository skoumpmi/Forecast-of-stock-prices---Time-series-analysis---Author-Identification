import numpy as np
import time
from datetime import datetime, timedelta
import os
import pandas as pd
from pmdarima.arima.utils import ndiffs
import pmdarima as pm
import cufflinks as cf
import configparser
import pickle
from pathlib import Path

config = configparser.ConfigParser()
config.read('config.ini')
signal_dir = os.path.join(os.getcwd(), "app", "signals", "ML_signals")

class ARIMA_Emergency:
    def ARIMA_train (self, train, asset):
        model = pm.auto_arima(train, start_p=1, start_q=1,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0,
                          D=1,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
        arima_dir = os.path.join(signal_dir, "ARIMA_models")
        if not os.path.exists(arima_dir):  # Creates the output directory if it does not exist
            os.makedirs(arima_dir)
        filename = os.path.join(arima_dir, '{}.pkl'.format(asset))
        pickle.dump(model, open(filename, 'wb'))

    def ARIMA_forecast(self, mydata, asset):
            cf.go_offline()
            cf.set_config_file(offline=False, world_readable=True)
            arima_dir = os.path.join(signal_dir, "ARIMA_models")
            filename = os.path.join(arima_dir, '{}.pkl'.format(asset['name']))
            model = pickle.load(open(filename, 'rb'))
            fc, confint = model.predict(n_periods=1, return_conf_int=True)
            predict = fc
            for i in range(0, len(predict)):
                mydata = mydata.reset_index()[['Date', 'Close']].append(
                    {'Date': datetime.today() + timedelta(days=i+1), 'Close': predict[i]}, ignore_index=True)
            mydata = mydata.set_index('Date')
            mean = mydata['Close'].mean()
            mydata['Var_ARIMA'] = 0.0
            var_list = []
            for m in range(0, len(mydata)):
                var_list.append(pow((mydata.iloc[m]['Close'] - mean), 2))
            mydata['Var_ARIMA'] = var_list
            var_mean = mydata['Var_ARIMA'].mean()
            mydata['Emer_ARIMA'] = 0.0
            Emer_list = []
            for j in range(0, len(mydata)):
                if abs(mydata.iloc[j]['Var_ARIMA']) > 1.5 * abs(var_mean):
                    Emer_list.append(1.0)
                else:
                    Emer_list.append(0.0)
            mydata['Emer_ARIMA'] = Emer_list
            mydata['Return_ARIMA'] = 0.0
            return_ARIMA_list = [0.0]
            for q in range(0, len(mydata) - 1):
                    return_ARIMA_list.append(
                        (mydata.iloc[q + 1]['Close'] - mydata.iloc[q]['Close']) / mydata.iloc[q]['Close'])
            mydata['Return_ARIMA'] = return_ARIMA_list
            std_ARIMA = mydata['Return_ARIMA'].std()
            mydata['Emerg_std_ARIMA'] = 0.0
            mydata.index = pd.to_datetime(mydata.index)
            Emeg_ARIMA_list_var = []
            for l in range(0, len(mydata)):
                if (abs(mydata.iloc[j]['Return_ARIMA'])) > 3 * std_ARIMA:
                    Emeg_ARIMA_list_var.append(1.0)
                else:
                    Emeg_ARIMA_list_var.append(0.0)
            mydata['Emerg_std_ARIMA'] = Emeg_ARIMA_list_var
            mydata = mydata[['Emer_ARIMA','Emerg_std_ARIMA']].reset_index(drop=True)
            return mydata


