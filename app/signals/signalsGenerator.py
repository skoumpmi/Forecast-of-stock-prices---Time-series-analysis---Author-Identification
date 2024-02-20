from app.signals.ML_signals.post_processing_signals import post_process_ML_signals
from app.signals.ML_signals.ARIMA_Alarm import ARIMA_Emergency
from app.signals.ML_signals.Facebook_Prophet_Alarm import Facebook_Prophet_Emergency
from app.signals.ML_signals.Random_Forest_Alarm import random_forest_alarm
from app.signals.HardAlarmSignals.Hard_Alarm_Estimation import Hard_Alarm
from app.signals.Mixed_Signals.Generate_Mixed_Signals import MixedSignalGenerator
import configparser
from datetime import datetime, timezone, timedelta
from app.signals.ML_signals.utils import Estimate_financial_indices
import pandas as pd
import pytz
import time
from app.database.dbManager import DBManager
from app.utilities.generalUtils import GeneralUtils
from app.sentiment.SentimentAnalysis import SentimentAnalysis

start_time = time.time()

# This class produces ML signals, Hard and Soft Emergency Alarms, Mixed Signals
class SignalsGenerator:

    def __init__(self):
        self.dbManager = DBManager()
        self.gu = GeneralUtils()
        self.sa = SentimentAnalysis()
        self.hm = Hard_Alarm()
        self.ms = MixedSignalGenerator()
        self.post_process_ML_sigs = post_process_ML_signals()
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

    def get_indeces_data(self, asset, days):
        latest_indeces_date = datetime.strptime(self.gu.convertMillisecsToDate(self.dbManager.get_latest_indeces_date()['latest_date'][0]), '%Y-%m-%d')
        start = self.gu.convertDateToMilliSecs((latest_indeces_date - timedelta(days=days)).date().strftime("%d/%m/%Y %H:%M:%S"))
        end = self.gu.getCurrentTimestsamp()
        indices = self.dbManager.get_asset_indices(asset['asset_id'], start, end)['asset_indices']
        data = pd.DataFrame(indices)
        data['date'] = data['date'].apply(self.gu.convertMillisecsToDate)
        data = data.rename(columns={"high": "High", "low": "Low", "open": "Open", "close": "Close", "volume": "Volume", "date": "Date"}).set_index("Date")
        return data

    def ml_train(self, asset):
        train_df = self.get_indeces_data(asset, 400)
        # TRAIN OF RANDOM FOREST ALARM SIGNALS
        rf_alarm = random_forest_alarm()
        rf_alarm.random_forest_train(data=train_df, asset=asset)  # initial_train ['yahoo_ticker']

        # TRAIN OF ARIMA SIGNLAS
        arima = ARIMA_Emergency()
        arima.ARIMA_train(train=train_df['Close'], asset=asset['name'])

        # TRAIN OF FACEBOOK PROPHET SIGNALS
        fb_prophet_emergency = Facebook_Prophet_Emergency()
        fb_prophet_emergency.Facebook_Prophet_train(train_df, False, asset['name'])

    def final_retrain(self, asset, days):
        latest_indeces_date = datetime.strptime(self.gu.convertMillisecsToDate(self.dbManager.get_latest_indeces_date()['latest_date'][0]), '%Y-%m-%d')
        start_date = self.gu.convertDateToMilliSecs((latest_indeces_date - timedelta(days=days)).date().strftime("%d/%m/%Y %H:%M:%S"))
        end_date = self.gu.getCurrentTimestsamp()
        self.post_process_ML_sigs.retrain(asset=asset, start=start_date, end=end_date)

    def indices_estimation(self, asset, start, end):
        obj = Estimate_financial_indices.estimate_indices()
        data_set = obj.estimate_final_indices(asset=asset, start=start, end=end)
        new_df = data_set.fillna(0).dropna().reset_index(drop=True)
        new_df.to_csv('{}.csv'.format(asset))

# Tha treksei mia fora gia na dimiourgisei to neuroniko
    def train_ml_signals(self):
        assets = self.dbManager.get_all_assets()
        for asset in assets:
            self.post_process_ML_sigs.activate_train(asset)

# Tha trexei mia fora tis X meres gia na ananewnei to modelo
    def retrain_ml_signals(self):
        assets = self.dbManager.get_all_assets()
        for asset in assets:
            self.final_retrain(asset, 110)

# Tha treksei mia fora gia na dimiourgisei to mixed modelo
    def train_mixed_signals(self):
        self.ms.train()

# Tha trexei mia fora tis X meres gia na ananewnei to modelo
    def train_emergencies(self):
        assets = self.dbManager.get_all_assets()
        for asset in assets:
            self.ml_train(asset)

# paragei ML-signals soft kai hard emergencies
    def generateMLSignalsAndEmergencies(self):
        signals = []
        alarms = []
        assets = self.dbManager.get_all_assets()
        current = self.gu.getCurrentTimestsamp()
        for asset in assets:
            start = self.dbManager.get_latest_indeces_date()['latest_date'][0] - self.gu.convertDaysToMiliseconds(111)
            sigs = self.post_process_ML_sigs.export_ML_signals(asset=asset, start=start, end=current)  # INFERENCE

            if sigs[0] == 1 or sigs[0] == 2:
			
                signals.append({'yahoo_ticker': asset['yahoo_ticker'], 'probability': float(sigs[2]), 'action': int(sigs[0]), 'type': 3, 'date': current})

            soft_alarm = self.post_process_ML_sigs.export_Emergency_Soft_Signal(asset=asset, final_frame=sigs[1])
            if soft_alarm == 1:
                alarms.append({'asset_id': asset['asset_id'], 'alarm': 1, 'date': current})

            df = self.get_indeces_data(asset, 365)
            hard_alarm = self.hm.emergency_data(df)
            if hard_alarm == 2:
                alarms.append({'asset_id': asset['asset_id'], 'alarm': 2, 'date': current})

        self.dbManager.insert_signals(signals)
        self.dbManager.insert_emergencies(alarms)

    def generateMixedSignals(self):
        mixed_signals = self.ms.inference()
        self.dbManager.insert_signals(mixed_signals)

    def generateSentimentSignals(self):
        try:
            self.sa.generate_sentiment_signal_for_all_assets()
        except Exception as e:
            print("--------------" + str(e) + "------------------")
        