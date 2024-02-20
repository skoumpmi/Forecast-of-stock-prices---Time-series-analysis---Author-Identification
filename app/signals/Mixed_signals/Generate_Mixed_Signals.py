import datetime
from datetime import timedelta, timezone, datetime
import pandas as pd
import holidays
import configparser
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import pickle
from app.database.dbManager import DBManager
from app.utilities.generalUtils import GeneralUtils

mixed_sigs_data_dir = os.path.join(os.getcwd(), "app", "signals", "Mixed_Signals", "Mixed_models")
config = configparser.ConfigParser()
config.read('config.ini')
dbManager = DBManager()


class mixed_signals:

    def __init__(self):
        self.dbManager = DBManager()
        self.gu = GeneralUtils()

    def is_unique(self, s):
        a = s.to_numpy()  # s.values (pandas<0.24)
        return (a[0] == a).all()

    def moving_average(self, df, n, feature):
        """Calculate the moving average for the given data.
        :param df: pandas.DataFrame
        :param n: timestep of averaging
        :return: pandas.DataFrame
        """
        MA = pd.Series(df['{}'.format(feature)].rolling(n, min_periods=n).mean())  # , name='MA_' + str(n)
        return MA

    def holiday_days(self):
        us_holidays = []
        for date in holidays.UnitedStates(years=datetime.now().year).items():
            us_holidays.append((str(date[0])))
        return us_holidays

    def fill_with_zeros(self, data, asset, feature_prefix, min_date):
        new_data = pd.DataFrame(columns=['Date', 'Asset', 'Signal'])
        base = datetime.today().date()
        delta = base - min_date
        date_data = pd.DataFrame(columns=['Date'])
        for i in range(delta.days + 1):
            day = min_date + timedelta(days=i)
            if (day.strftime('%Y-%m-%d') not in self.holiday_days()) and (pd.to_datetime(day).weekday() < 5):
                date_data = date_data.append({'Date': day}, ignore_index=True)
        for j in range(0, len(date_data)):
            if (any(data["Date"][data["Date"] == date_data['Date'][j]])):
                if (data[data['Date'] == (data["Date"][data["Date"] == date_data['Date'][j]]).values[0]]['Asset'].values[0]) == asset:
                    new_data = new_data.append(
                        {'Date': data[data['Date'] == (data["Date"][data["Date"] == date_data['Date'][j]]).values[0]]['Date'].values[0],
                         'Asset': '{}_{}'.format(feature_prefix, asset),
                         'Signal': data[data['Date'] == (data["Date"][data["Date"] == date_data['Date'][j]]).values[0]]['Signal'].values[0]},
                        ignore_index=True)
                else:
                    new_data = new_data.append({'Date': date_data['Date'][j], 'Asset': '{}_{}'.format(feature_prefix, asset), 'Signal': 0},
                                               ignore_index=True)
            else:
                new_data = new_data.append({'Date': date_data['Date'][j], 'Asset': '{}_{}'.format(feature_prefix, asset), 'Signal': 0},
                                           ignore_index=True)
        return new_data

    def create_train_data(self, df):
        data = df.copy()
        grouped = data.groupby('Date')
        data_sig = pd.DataFrame()
        for item in grouped:
            # We want each row to represent a diffrent date
            data_sig = pd.concat(
                [data_sig, item[1].set_index('Asset')['Signal'].to_frame().transpose()])
            data_sig = data_sig.reset_index(drop=True)
        return data_sig

    def meta_process_data(self, asset, initial_concat, length):
        indices = self.dbManager.get_asset_indices(asset['asset_id'], None, None)['asset_indices']
        data = pd.DataFrame(indices)
        data['date'] = data['date'].apply(self.gu.convertMillisecsToDate)
        close_df = data.rename(columns={"high": "High", "low": "Low", "open": "Open",
                                        "close": "Close", "volume": "Volume", "date": "Date"}).set_index("Date").iloc[-length:]
        return_list = [0.0]
        for j in range(0, len(close_df) - 1):
            return_value = np.log(close_df.iloc[j + 1]['Close']) - np.log(close_df.iloc[j]['Close'])
            return_list.append(return_value)
        close_df['Return'] = return_list
        for i in range(0, len(close_df)):
            close_df['MA_slow'] = self.moving_average(close_df['Return'].to_frame(), 50, 'Return')
            close_df['MA_fast'] = self.moving_average(close_df['Return'].to_frame(), 5, 'Return')
        close_df = close_df.fillna(.0)
        final_frame_real_list = []
        for n in range(0, len(close_df)):
            if (close_df.iloc[n]['MA_fast'] > close_df.iloc[n]['MA_slow'] + 0.02) and (
                    close_df.iloc[n]['MA_slow'] != 0):
                final_frame_real_list.append(1)
            elif (close_df.iloc[n]['MA_fast'] < close_df.iloc[n]['MA_slow'] - 0.02) and (
                    close_df.iloc[n]['MA_slow'] != 0):
                final_frame_real_list.append(2)
            else:
                final_frame_real_list.append(0)

        close_df['{}'.format(asset['name'])] = final_frame_real_list
        close_df = close_df['{}'.format(asset['name'])].to_frame().reset_index(drop=True)
        initial_concat = pd.concat([initial_concat, close_df], axis=1)
        return initial_concat

    def train_models(self, train_data, asset):
        # TRAIN OF MODELS PROCEDURE
        X = train_data.iloc[:, train_data.columns != '{}'.format(asset['name'])]
        y = train_data.loc[:, train_data.columns == '{}'.format(asset['name'])]
        if not os.path.exists(mixed_sigs_data_dir):  # Creates the output directory if it does not exist
            os.makedirs(mixed_sigs_data_dir)
        if self.is_unique(y) == False:
            logreg = LogisticRegression(penalty='l1', solver='saga')  # LR algorithm
            model = logreg.fit(X, y)
            filename = os.path.join(mixed_sigs_data_dir, '{}.pkl'.format(asset['name']))
            pickle.dump(model, open(filename, 'wb'))

    def mixed_infernece(self, import_data, asset):
        X = import_data.iloc[:, import_data.columns != '{}'.format(asset['name'])]
        filename = os.path.join(mixed_sigs_data_dir, '{}.pkl'.format(asset['name']))
        if os.path.isfile(filename):
            model = pickle.load(open(filename, 'rb'))
            y_pred = model.predict(X)[-1]
            probs = model.predict_proba(X)
            probs = (round(max(probs[-1]), 3)) * 100
        else:
            y_pred = 0
            probs = 33.0
        return y_pred, probs


class MixedSignalGenerator():

    def __init__(self):
        self.dbManager = DBManager()
        self.gu = GeneralUtils()
        self.ms = mixed_signals()

    def get_indeces_data(self, asset, days):
        latest_indeces_date = datetime.strptime(self.gu.convertMillisecsToDate(self.dbManager.get_latest_indeces_date()['latest_date'][0]),
                                                '%Y-%m-%d')
        start = self.gu.convertDateToMilliSecs((latest_indeces_date - timedelta(days=days)).date().strftime("%d/%m/%Y %H:%M:%S"))
        end = self.gu.getCurrentTimestsamp()
        indices = self.dbManager.get_asset_indices(asset['asset_id'], start, end)['asset_indices']
        data = pd.DataFrame(indices)
        data['date'] = data['date'].apply(self.gu.convertMillisecsToDate)
        data = data.rename(columns={"high": "High", "low": "Low", "open": "Open", "close": "Close", "volume": "Volume", "date": "Date"}).set_index(
            "Date")
        return data

    def get_train_data(self, assets):
        data = pd.DataFrame(dbManager.get_all_signals(signal_types=[1, 2, 3]))
        data['Date'] = data['Date'].apply(lambda x: datetime.fromtimestamp(x/1000.0).date())
        ml_data = data.loc[data['Type'] == 3].reset_index(drop=True)
        sent_data = data.loc[data['Type'] == 2].reset_index(drop=True)
        tech_data = data.loc[data['Type'] == 1].reset_index(drop=True)

        ml_init = pd.DataFrame()
        sent_init = pd.DataFrame()
        tech_init = pd.DataFrame()

        sent_min = sent_data['Date'].min()
        tech_min = tech_data['Date'].min()
        ml_min = ml_data['Date'].min()
        min_date = min([sent_min, tech_min, ml_min])
        post_process_data = pd.DataFrame()
        ms = mixed_signals()
        for asset in assets:

            sent_init = pd.concat([sent_init, ms.fill_with_zeros(data=sent_data, asset=asset['name'], feature_prefix='sent', min_date=min_date)])
            tech_init = pd.concat([tech_init, ms.fill_with_zeros(data=tech_data, asset=asset['name'], feature_prefix='tech', min_date=min_date)])
            ml_init = pd.concat([ml_init, ms.fill_with_zeros(data=ml_data, asset=asset['name'], feature_prefix='ml', min_date=min_date)])
            grouped = sent_init.groupby('Asset')
            for item in grouped:
                length = len(item[1])
                break
            post_process_data = ms.meta_process_data(asset=asset, initial_concat=post_process_data, length=length)
        sent_final = ms.create_train_data(sent_init)
        tech_final = ms.create_train_data(tech_init)
        ml_final = ms.create_train_data(ml_init)
        final_data = pd.concat([sent_final, tech_final, ml_final, post_process_data], axis=1)

        return final_data

    def train(self):
        assets = dbManager.get_all_assets()
        train_data = self.get_train_data(assets)
        for asset in assets:
            self.ms.train_models(train_data, asset)  # TRAIN oF MIXED MODELS

    def inference(self):
        assets = dbManager.get_all_assets()
        train_data = self.get_train_data(assets)
        mixed = []
        for asset in assets:
            result, possibility = self.ms.mixed_infernece(train_data, asset)
            if result == 1 or result == 2:
                i = 1
                while ((datetime.now(timezone.utc).date() + timedelta(days=i)).strftime(
                        '%Y-%m-%d') not in self.ms.holiday_days()) and (
                        pd.to_datetime(datetime.now(timezone.utc).date() + timedelta(days=i)).weekday() >= 5):
                    i += 1
                if possibility < 55.0:
                    possibility = 55.0
                elif possibility > 75:
                    possibility = 75.0
                timestamp = int((datetime.now(timezone.utc) + timedelta(days=i)).timestamp() * 1000)
                mixed.append({'yahoo_ticker': asset['yahoo_ticker'], 'probability': float(possibility/100), 'action': result, 'type': 4, 'date': timestamp})
        return mixed
