import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import os
from app.signals.ML_signals.utils import Estimate_financial_indices
import tensorflow as tf
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import configparser
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

config = configparser.ConfigParser()
config.read('config.ini')
signal_dir = os.path.join(os.getcwd(), "app", "signals", "ML_signals")

class ML_signals():
    def moving_average(self, df, n, feature):
        """Calculate the moving average for the given data.
        :param df: pandas.DataFrame
        :param n: timestep of averaging
        :return: pandas.DataFrame
        """
        MA = pd.Series(df['{}'.format(feature)].rolling(n, min_periods=n).mean())
        return MA

    def transform_to_signals(self, mydata):
        mydata['Return_Deep'] = 0.0
        mean = mydata['Close'].mean()
        mydata['Var_Deep'] = 0.0
        var_list = []
        for i in range(0, len(mydata)):
            var_list.append(pow((mydata.iloc[i]['Close'] - mean),2))
        mydata['Var_Deep'] = var_list
        mydata['Emer_Deep'] = 0.0
        pos_list = []
        for m in range(0, len(mydata)):
            if abs(mydata.iloc[m]['Close']) > abs(mean):
                pos_list.append(1.0)
            else:
                pos_list.append(0.0)
        mydata['Emer_Deep'] = pos_list
        return_deep_list = [0.0]
        for i in range(0, len(mydata) - 1):
            return_deep_list.append(
                (mydata.iloc[i + 1]['Close'] - mydata.iloc[i]['Close']) / mydata.iloc[i]['Close'])
        mydata['Return_Deep'] = return_deep_list
        std_deep = mydata['Return_Deep'].std()
        mydata['Emerg_std_Deep'] = 0.0
        mydata.index = pd.to_datetime(mydata.index)
        Emeg_std_list = []
        for j in range(0, len(mydata)):
            if (abs(mydata[mydata['Return_Deep'] == mydata.iloc[j]['Return_Deep']]['Return_Deep'][0]) > 2 * std_deep):
                Emeg_std_list.append(1.0)
            else:
                Emeg_std_list.append(0.0)
        mydata['Emerg_std_Deep'] = Emeg_std_list
        return mydata[['Emer_Deep','Emerg_std_Deep']]

    def train(self, windowsize, timestep, numepochs, asset):
        obj = Estimate_financial_indices.estimate_indices()
        data_set = obj.estimate_final_indices(asset=asset, start=None, end=None)
        df = data_set.copy()
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        df = df.dropna().reset_index(drop=True)

        window_size: object = windowsize
        time_step = timestep
        num_features = len(df.columns) -1
        num_df = df.to_numpy()
        res_x, res_y = [], []
        for i in range(len(df) - window_size - 1):
            res_x.append(num_df[i:i + window_size, : len(df.columns) -1])

            res_y.append(np.array(num_df[(i + window_size + 1):(i + window_size + time_step + 1), len(df.columns) -1]))
        res_x = res_x[:len(res_x) - time_step]
        res_y = res_y[:len(res_y) - time_step]
        res_x2 = np.asarray(res_x)
        res_x2 = res_x2.reshape(res_x2.shape[0], res_x2.shape[1], num_features)
        res_y2 = np.vstack(res_y)
        res_y2 = res_y2.reshape(res_y2.shape[0], res_y2.shape[1], 1)
        train_x, test_x, train_y, test_y = train_test_split(res_x2, res_y2, shuffle=False, test_size=5)
        n_filters = 64
        filter_width = 2
        dilation_rates = [2 ** i for i in range(8)]

        history_seq = Input(shape=(windowsize, len(df.columns) -1))
        x = history_seq
        for dilation_rate in dilation_rates:
            x = Conv1D(filters=n_filters,
                       kernel_size=filter_width,
                       padding='causal',
                       dilation_rate=dilation_rate)(x)
        x = Dense(40, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        x = Dropout(.1)(x)
        x = Dense(1)(x)
        def slice(x, seq_length):
            return x[:, -seq_length:, :]
        pred_seq_train = Lambda(slice, arguments={'seq_length': time_step})(x)
        model = Model(history_seq, pred_seq_train)
        batch_size = 64
        epochs = numepochs
        model.compile(Adam(), loss='mean_squared_error', )
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
        history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y), callbacks=[es])

        output_dir = os.path.join(signal_dir, "signal_models", asset['name'])
        output_file_path = os.path.join(output_dir, asset['name'] + '.pkl')
        f = open(output_file_path, "wb")
        pickle.dump(history.history, f)
        f.close()
        model.save(output_dir)


    def inference(self, df1, asset, start, end):
        obj = Estimate_financial_indices.estimate_indices()
        data_set = obj.estimate_final_indices(asset, start=start, end=end)
        new_df = data_set.fillna(0).dropna().reset_index(drop=True)
        df = new_df.copy()
        window_size = 64
        time_step = 1
        num_features = len(df.columns) - 1
        model3 = tf.keras.models.load_model(os.path.join(signal_dir, "signal_models", asset['name']))
        num_df = df.to_numpy()
        res_x, res_y = [], []

        for i in range(len(df) - window_size - 1):
            res_x.append(num_df[i:i + window_size, :num_features])

            res_y.append(np.array(num_df[(i + window_size + 1):(i + window_size + time_step + 1), num_features]))
        res_x = res_x[:len(res_x) - time_step]
        res_x2 = np.asarray(res_x)
        res_x2 = res_x2.reshape(res_x2.shape[0], res_x2.shape[1], num_features)
        preds = model3.predict(res_x2)
        predict = preds[-1]
        min_value = df1['Close'][new_df[new_df['close'] == new_df['close'].min()].index.values[0]]
        max_value = df1['Close'][new_df[new_df['close'] == new_df['close'].max()].index.values[0]]
        new_predict = (predict * max_value + (1 - predict) * min_value)
        
        new_preds = [(item *  max_value + (1 - item) * min_value) for item in preds]
        new_real = df1.iloc[-len(new_preds):]['Close'].values.tolist()
        new_preds = [item for sublist in np.vstack(new_preds) for item in sublist]
        final_pred = [(((item) - min(new_preds+new_real))/(max(new_preds+new_real)-min(new_preds+new_real))) for item in new_preds]
        final_real = [(((item) - min(new_preds+new_real))/(max(new_preds+new_real)-min(new_preds+new_real))) for item in new_real]
        mse = 1 - mean_squared_error(final_real, final_pred)

        close_df = df1['Close'].to_frame()
        close_df = close_df.append({'Close': new_predict[0][0]}, ignore_index=True)
        return_list = [0.0]
        for j in range(0, len(close_df) - 1):
            return_value = np.log(close_df.iloc[j + 1]['Close']) - np.log(close_df.iloc[j]['Close'])
            return_list.append(return_value)
        close_df['Return'] = return_list
        
        final_frame_real_list = []
        mean = close_df['Return'].mean()
        std = close_df['Return'].std()
        possibility = 0.0
        for n in range(0, len(close_df)):
            if ((close_df.iloc[n]['Return']) > (mean + 3 * (std))) and mse >= 0.55 and mse <= 0.75:
                possibility = round(mse, 3)
                final_frame_real_list.append(1)
            elif ((close_df.iloc[n]['Return'])  < (mean) - (3*(std))) and mse >= 0.55 and mse <= 0.75:
                final_frame_real_list.append(2)
                possibility = round(mse, 3)
            else:
                final_frame_real_list.append(0)
        close_df['Signal'] = final_frame_real_list
        
        return close_df, possibility




