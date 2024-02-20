from app.signals.ML_signals.mlSignals import ML_signals
import pandas as pd
import os
import configparser
from pathlib import Path
import time
from datetime import datetime, timedelta
from app.signals.ML_signals import Random_Forest_Alarm, Facebook_Prophet_Alarm, ARIMA_Alarm
import numpy as np
import holidays
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from app.signals.ML_signals.utils import Estimate_financial_indices
from app.database.dbManager import DBManager
from app.utilities.generalUtils import GeneralUtils

signal_dir = os.path.join(os.getcwd(), "app", "signals", "ML_signals")

class post_process_ML_signals:

    def __init__(self):
        self.dbManager = DBManager()
        self.gu = GeneralUtils()
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.ml_signals = ML_signals()

    def export_holiday_days(self):
        us_holidays = []
        for date in holidays.UnitedStates(years=datetime.now().year).items():
            us_holidays.append((str(date[0])))
        return us_holidays

    def activate_train(self, asset):
        output_dir = os.path.join(signal_dir, "signal_models", asset['name'])
        if not os.path.exists(output_dir):  # Creates the output directory if it does not exist
            os.makedirs(output_dir)
        self.ml_signals.train(windowsize=64, timestep=1, numepochs=200, asset=asset)


    def export_ML_signals(self, asset, start, end):
        dbManager = DBManager()
        indices = dbManager.get_asset_indices(asset['asset_id'], start, end)['asset_indices']
        df = pd.DataFrame(indices)
        df = df.rename(columns={"close": "Close", "date": "Date"})[['Date', 'Close']]
        result = self.ml_signals.inference(df, asset=asset, start=start, end=end)

        final_frame = result[0]

        date = df['Date'].to_frame()
        gu = GeneralUtils()
        tomorrow = datetime.today() + timedelta(days=1)
        date['Date'] = date['Date'].apply(gu.convertMillisecsToDate)
        date = date.append({'Date': tomorrow}, ignore_index=True)
        ML_frame = pd.concat([date, final_frame], axis=1).set_index('Date')
        prediction = ML_frame.iloc[-1, -1]
        new_final_frame = self.ml_signals.transform_to_signals(ML_frame).reset_index(drop=True)
        possibility = result[1]

        return prediction, new_final_frame, possibility

    def get_indeces_data(self, asset, days):
        dbManager = DBManager()
        gu = GeneralUtils()
        latest_indeces_date = datetime.strptime(gu.convertMillisecsToDate(dbManager.get_latest_indeces_date()['latest_date'][0]), '%Y-%m-%d')
        start = gu.convertDateToMilliSecs((latest_indeces_date - timedelta(days=days)).date().strftime("%d/%m/%Y %H:%M:%S"))
        end = gu.getCurrentTimestsamp()
        indices = dbManager.get_asset_indices(asset['asset_id'], start, end)['asset_indices']
        data = pd.DataFrame(indices)
        data['date'] = data['date'].apply(gu.convertMillisecsToDate)
        data = data.rename(columns={"high": "High", "low": "Low", "open": "Open", "close": "Close", "volume": "Volume", "date": "Date"}).set_index("Date")
        return data

    def export_Emergency_Soft_Signal(self, asset, final_frame):
        data = self.get_indeces_data(asset, 400)
        rf_data = Random_Forest_Alarm.random_forest_alarm().random_forest_inference(data=data, asset=asset)
        arima_data = ARIMA_Alarm.ARIMA_Emergency().ARIMA_forecast(mydata=self.get_indeces_data(asset, 400), asset=asset)
        prophet_data = Facebook_Prophet_Alarm.Facebook_Prophet_Emergency().Facebook_Prophet_Emergency_Estimation(asset)
        final_estimator = pd.concat(
            [final_frame.iloc[[-1]].reset_index(drop=True), rf_data.iloc[[-1]].reset_index(drop=True),
             arima_data.iloc[[-1]].reset_index(drop=True),
             prophet_data.iloc[[-1]].reset_index(drop=True)], axis=1)
        k = final_estimator.sum(axis=1).to_frame()
        k.columns = ['sum']
        k['sig'] = 0
        if (k.iloc[0]['sum']) / 8 > 0.5:
            k['sig'][0] = 1

        return k['sig'][0]

    def retrain(self, asset, start, end):
        batch_size = 64
        epochs = 200
        obj = Estimate_financial_indices.estimate_indices()
        data_set = obj.estimate_final_indices(asset=asset, start=start, end=end)
        new_df = data_set.fillna(0).dropna().reset_index(drop=True)
        df = new_df.copy()
        window_size = 64
        time_step = 1
        num_features = len(df.columns) - 1
        model3 = tf.keras.models.load_model(os.path.join(signal_dir, "signal_models", asset['name']))
        num_df = df.to_numpy()
        res_x, res_y = [], []
        for i in range(len(df) - window_size):

            res_x.append(num_df[i:i + window_size, :num_features])

            res_y.append(np.array(num_df[(i + window_size + 1):(i + window_size + time_step + 1), num_features]))
        res_x = res_x[:len(res_x) - time_step]
        res_y = res_y[:len(res_y) - time_step]
        res_x2 = np.asarray(res_x)
        res_y2 = np.vstack(res_y)
        res_y2 = res_y2.reshape(res_y2.shape[0], res_y2.shape[1], 1)
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
        model3.fit(res_x2, res_y2, batch_size=batch_size, epochs=epochs,  callbacks=[es])
        output_dir = os.path.join(signal_dir, "signal_models", asset['name'])
        model3.save(output_dir)
