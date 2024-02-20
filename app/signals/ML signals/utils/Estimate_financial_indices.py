import pandas as pd
import numpy as np
from fancyimpute import KNN
from pandas_datareader import data as pdr
from finta import TA
from sklearn.linear_model import LinearRegression
import datetime
import time
from datetime import datetime, timezone
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import statistics
from sklearn.pipeline import Pipeline
from pyti.detrended_price_oscillator import detrended_price_oscillator as DPO
from pyti.price_oscillator import price_oscillator as POSC
from pyti.standard_deviation import standard_deviation as STDDV
from pyti.vertical_horizontal_filter import vertical_horizontal_filter as VHF
from pyti.volume_oscillator import volume_oscillator as VO
from pyti.volatility import volatility as VOLAT
import os
import configparser
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from ....database.dbManager import DBManager
from ....utilities.generalUtils import GeneralUtils

config = configparser.ConfigParser()
config.read('config.ini')

signal_dir = os.path.join(os.getcwd(), "app", "signals", "ML_signals")

class estimate_indices():
    def ao_oscillator(self,df):#Accumulation / Distribution oscillator
        """Calculate the Accumulation / Distribution oscillator for given data.
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        k=df['Close'].diff()
        ao_oscillator = pd.Series((df['low'] - k) / (df['high'] - df['low']), name='ao_osc%k')
        return ao_oscillator

    def stoh_D_slow(self,df, n):
        """Calculate the Slow Stohastic oscillator for given data.
        :param df: pandas.DataFrame
        :param n:
        :return: pandas.DataFrame
        """
        D_slow = pd.Series(df['%D'].rolling(n, min_periods=n).mean(), name='%D_slow' + str(n))
        return D_slow
    def moving_average(self,df, n):
        """Calculate the moving average for the given data.
        :param df: pandas.DataFrame
        :param n: timestep of averaging
        :return: pandas.DataFrame
        """
        MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
        return MA

    def moving_variance(self,df, n):
        """Calculate the moving variance for the given data.
        :param df: pandas.DataFrame
        :param n: timestep of averaging
        :return: pandas.DataFrame
        """
        MV = pd.Series((df['Close']/df['Close'].mean()).rolling(n, min_periods=n).mean(), name='MA_' + str(n))
        return MV

    def vol_moving_average(self,df, n):
        """Calculate the moving volume average for the given data.

        :param df: pandas.DataFrame
        :param n: timestep of averaging
        :return: pandas.DataFrame
        """
        VMA = pd.Series(df['volume'].rolling(n, min_periods=n).mean(), name='VMA_' + str(n))
        return VMA

    def Disp_n(self,df, n):
        """Calculate the disparity index for given data.
        :param df: pandas.DataFrame
        :param n: timestep of Disparity
        :return: pandas.DataFrame
        """
        Disp_n = pd.Series((df['Close'] / self.moving_average(df, n)), name='Disp%n')
        return Disp_n

    def PSY(self,df, n):
        """Calculate PSYn is the ratio of the number of rising periods over the n day period for given data.
        Based on Qiu, M., & Song, Y. (2016). Predicting the direction of stock market index movement using an optimized artificial neural network model. PloS one, 11(5), e0155133
        the estimation of PSY is done for 12 days.
        :param df: pandas.DataFrame
        :param n: timestep of PSY estimator.
        :return: pandas.DataFrame
        """
        i = 0
        UpI = 0
        DoI = [0]
        for i in range(0, len(df) - 1):

            UpMove = df.loc[i + 1, 'close'] - df.loc[i, 'close']
            if UpMove > 0:
                UpI = 1
            else:
                UpI = 0

            DoI.append(UpI)
        DoI = pd.Series(DoI)

        PosDI = pd.Series(DoI.rolling(n, min_periods=n).mean(), name='PSY_12' + str(n))
        return PosDI
    def OSCP(self,df, short_period, long_period):
        """Calculate price oscillator OSCP for given data.
        :param df: pandas.DataFrame
        :param short_period: number of timesteps which based the estimation of short-time moving average
        :param long_period: number of timesteps which based the estimation of long-time moving average
        :return: pandas.DataFrame
        """
        df = df.rename(columns={"Close": "close", "Low": "low", "High": "high", "Open": "open", "Volume": "volume"})
        OSCP = pd.Series(((TA.SMA(df,short_period) - TA.SMA(df,long_period)) / TA.SMA(df,short_period)), name='OSCP%n')
        return OSCP
    def BIAS(self,df, n):
        """Calculate the bias ratio for given data.
            :param df: pandas.DataFrame
            :param n: timestep of bias estimation. based on Qiu, M., & Song, Y. (2016). Predicting the direction of stock market index movement
            using an optimized artificial neural network model. PloS one, 11(5), e0155133 the n factor is set equal to 6.
            :return: pandas.DataFrame
        """
        BIAS = pd.Series((df['Close'] - self.moving_average(df, n)/self.moving_average(df, n)), name='Disp%n')
        return BIAS

    def rolling_variance(self,df , key):
        #It is an auxiliary function to estimate the next function moving_variance_ratio
        rolling_variance= pd.Series(df[key]).rolling(5).var()
        return rolling_variance

    def moving_variance_ratio(self,df):
        """Calculate the moving variance ratio (MVR). Based on Yu, L., Chen, H., Wang, S., & Lai, K. K. (2008). Evolving least squares support vector machines
            for stock market trend mining. IEEE transactions on evolutionary computation, 13(1), 87-102.
            It is estimated from formula MVt^2/MVt-m^2
            :param df: pandas.DataFrame
            :return: pandas.DataFrame
        """
        rolling_variance_ratio=[]
        for i in range(0, len(df) - 5):
            rolling_ratio = (df.loc[i + 5, 'close'])
            rolling_variance_ratio.append(rolling_ratio)
        rolling_variance_ratio = ((pow(self.rolling_variance(df, 'close'), 2)) / (pow (pd.Series(rolling_variance_ratio).rolling(5).var(), 2)))
        return rolling_variance_ratio

    def real_trend(self,df):
        """Calculate the trend of close price of  stocks.
            trend = 0 means the stock price is decreased in relation with the previous timestep
            trend = 1 means the stock price is rised in relation with previous timestep.
            :param df: pandas.DataFrame
            :return: pandas.DataFrame
            """
        i = 0
        UpI = 0
        DoI = [0]
        for i in range(0, len(df) - 1):

            UpMove = df.loc[i + 1, 'close'] - df.loc[i, 'close']
            if UpMove > 0:
                UpI = 1
            else:
                UpI = 0
            DoI.append(UpI)
        DoI = pd.Series(DoI)
        return DoI
    def linear_regress(self,df):
        #Estimate the forecating closing values of stocks from linear regression algorithms
        X = df.loc[:, 'close'].values.reshape(-1, 1)  # values converts it into a numpy array
        Y = df.loc[:, 'real'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X,Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)
        return(Y_pred)

    def ASY_1 (self, df):
        #Estimate the return value for 1 timestep
        df = df.rename(columns={"Close": "close", "Low": "low", "High": "high", "Open": "open", "Volume": "volume"})
        df['ASY1'] = (pd.Series((np.log(df['close'])).rolling(1, min_periods=1).mean(), name='ASY_' + str(1)))
        return df['ASY1']

    def ASY (self,df,k):
        ''' Estimate the mean return af k previous timestep.
        Based on Qiu, M., & Song, Y. (2016). Predicting the direction of stock market
        index movement using an optimized artificial neural network model. PloS one, 11(5), e0155133 is estimated the ASY for 2 to 5 timestep.
        :param df: pandas.DataFrame
        :param k: number of timesteps for which the mean ASY is estimated
        :return: pandas.DataFrame
        '''
        number = 0.0
        df = df.rename(columns={"Close": "close", "Low": "low", "High": "high", "Open": "open", "Volume": "volume"})
        df['ASY{}'.format(k)] = 0.0
        for i in range(k,len(df)):
            for j in range(i-k, i):
                if j!=0:
                    number += (np.log(df['close'][j]) - np.log(df['close'][j - 1]))
                else:
                    number = 0.0
            df['ASY{}'.format(k)][i] = number/k
            number = 0.0
        return df['ASY{}'.format(k)]

    def VMACD(self,ohlc,period_fast,period_slow,signal,adjust) :
        """
        MACD, MACD Signal and MACD difference.
        The MACD Line oscillates above and below the zero line, which is also known as the centerline.
        These crossovers signal that the 12-day EMA has crossed the 26-day EMA. The direction, of course, depends on the direction of the moving average cross.
        Positive MACD indicates that the 12-day EMA is above the 26-day EMA. Positive values increase as the shorter EMA diverges further from the longer EMA.
        This means upside momentum is increasing. Negative MACD values indicates that the 12-day EMA is below the 26-day EMA.
        Negative values increase as the shorter EMA diverges further below the longer EMA. This means downside momentum is increasing.
        Signal line crossovers are the most common MACD signals. The signal line is a 9-day EMA of the MACD Line.
        As a moving average of the indicator, it trails the MACD and makes it easier to spot MACD turns.
        A bullish crossover occurs when the MACD turns up and crosses above the signal line.
        A bearish crossover occurs when the MACD turns down and crosses below the signal line.
        """
        EMA_fast = pd.Series(
            ohlc["volume"].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name="EMA_fast",
        )
        EMA_slow = pd.Series(
            ohlc["volume"].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
            name="EMA_slow",
        )
        MACD = pd.Series(EMA_fast - EMA_slow, name="MACD")
        MACD_signal = pd.Series(
            MACD.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name="SIGNAL"
        )

        return pd.concat([MACD, MACD_signal], axis=1)
    def fill_for_noncomputable_vals(self,input_data, result_data):
        non_computable_values = np.repeat(
            np.nan, len(input_data) - len(result_data)
            )
        filled_result_data = np.append(non_computable_values, result_data)
        return filled_result_data
    def standard_deviation(self,data, period):
        """
        Standard Deviation.
        Formula:
        std = sqrt(avg(abs(x - avg(x))^2))
        """
        #catch_errors.check_for_period_error(data, period)
        stds = [np.std(data[idx+1-period:idx+1], ddof=1) for idx in range(period-1, len(data))]
        stds = self.fill_for_noncomputable_vals(data, stds)
        return stds
    def estimate_Ta_attributes(self,attribute, df, function):
        r''' With this function we are going to estimate lots of indices.
        ROC: Price Rate of Change: The Price Rate of Change (ROC) is a momentum-based technical indicator that measures the percentage change in price
        between the current price and the price a certain number of periods ago.
        %K oscillator: Slow stochastic oscillator %K is a momentum indicator comparing a particular closing price
        of a security to a range of its prices over a certain period of time.
        %D oscillator: The "fast" stochastic indicator is taken as %D = 3-period moving average of %K.
        MOM: Momentum is the rate of acceleration of a security's price or volume—
        %R Williams: Williams %R, also known as the Williams Percent Range, is a type of momentum indicator
        that moves between 0 and -100 and measures overbought and oversold levels.
        EXPM: Exponential Moving Average: An exponential moving average (EMA) is a type of moving average (MA)
        that places a greater weight and significance on the most recent data points.
        CCI: Commodity Channel Index is a momentum-based oscillator used to help determine
        when an investment vehicle is reaching a condition of being overbought or oversold.
        RSI: Relative STrength index is a momentum indicator used in technical analysis that measures
        the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.
        OBV: On Balance Volume is a technical trading momentum indicator
        that uses volume flow to predict changes in stock price.
        ATR: Average True Range is a technical analysis indicator that measures market volatility
        by decomposing the entire range of an asset price for that period.
        TRIX: The Triple Exponential Average is a momentum indicator used by technical traders
        that shows the percentage change in a triple exponentially smoothed moving average.
        VAMA: The Volume Adjusted Moving Average is a special type Moving Average that takes into account not just the time period, but also the Volume of single days.
        SAR : The parabolic SAR indicator is used by traders to determine trend direction and potential reversals in price.
        '''
        df[attribute] = function(df)
        return df[attribute]

    def estimate_Ta_comb_attributes(self,df, function):
        r'''
        Estimate features from FINTA library that includes more than 1 feature.
        BBANDS: A Bollinger Band® is a technical analysis tool defined by a set of trendlines plotted two standard
        deviations (positively and negatively) away from a simple moving average (SMA) of a security's price, but which can be adjusted to user preferences.
        BBANDS_low & BBANDS_high: Many traders believe the closer the prices move to the upper(high) band, the more overbought the market, and the closer
        the prices move to the lower (low) band, the more oversold the market.
          DMI: Directional Movement Indicator identifies in which direction the price of an asset is moving.
        DMI_pos:  For the purpose of comparing prior highs DMI estimate the component positive DMI (+DI).
        DMI_neg: For the purpose of comparing prior lows DMI estimate the component negative DMI (-DI).
        PPO: Percentage Price Oscillator  is a technical momentum indicator that shows the relationship between two moving averages in percentage terms.
        PPO_sig : 9-period EMA of PPO and EMA=Exponential moving average
        PPO_hist: = PPO − PPO_sig
        '''
        return function(df)
    def estimate_pyti_attributes_one_arg(self,feature, df, timestep):
        r'''
        :param feature: the feature which get from pyti library
        :param df: pandas dataframe in which have the prices of stocks and concatenate the features
        :param timestep: rolling moving average window in which it will be estimated the indices
        :return: pandas dataframe with estimated feature
        VHF: Vertical Horizontal Filter identifys  trending and ranging markets. VHF measures the level of trend activity.
           VPT: The volume price trend indicator helps determine a security’s price direction and strength of price change.
        VOLAT:  Volatility is a statistical measure of the dispersion of returns for a given security or market index.
        DPO: The detrended price oscillator is an oscillator that strips out price trends in an effort to estimate the length of price cycles from peak to peak or trough to trough.
        STDDV: Standard Deviation is a statistic that measures the dispersion of a dataset relative to its mean and is calculated as the square root of the variance.
        '''
        return feature(df.close.to_numpy(), timestep)
    def estimate_pyti_attributes_two_arg(self,feature, df, timestep_fast, timestep_slow):
        r'''
        :param feature: the feature which get from pyti library
        :param df: pandas dataframe in which have the prices of stocks and concatenate the features
        :param timestep_fast: the short-time rolling moving average window in which it will be estimated the indices
        :param timestep_slow: the long-time rolling moving average window in which it will be estimated the indices
        :return: pandas dataframe with estimated feature
        POSC:The Price Oscillator is a technical indicator that measures
        whether or not the most recent closing price is above or below the preceding closing price.
        VO: Volume Oscillator  measures volume by measuring the relationship between two moving averages.
        The volume oscillator indicator calculates a fast and slow volume moving average.
        '''
        return feature(df.close.to_numpy(), timestep_fast, timestep_slow)
    def minmax_scale(self,df):
        frames = []
        for x in df:
            transformer = MinMaxScaler().fit(df[x].values.reshape(-1, 1))
            df[x] = transformer.transform(df[x].values.reshape(-1, 1))
            frames.append(df[x])
        frames = pd.DataFrame(frames).transpose()
        return frames
    def make_minmax_dataset(self,df):
        feat_list = []
        for x in df:
            feat_list.append(x)
        feat_list = feat_list [1:]
        normalised= ['ATR','plus','minus','VHF','STDDV','LR']
        unormalised = [item for item in feat_list if item not in normalised]
        unormalised_frames = []
        for x in unormalised:
            unormalised_frames.append(df[x])
        unormalised_frames = pd.DataFrame(unormalised_frames).transpose()
        unormalised_frames = unormalised_frames.replace(np.inf, np.nan)
        unormalised_frames = self.minmax_scale(unormalised_frames)
        normalised_frames = []
        for x in normalised:
            normalised_frames.append(df[x])
        normalised_frames = pd.DataFrame(normalised_frames).transpose()
        final_frames=[unormalised_frames.transpose(),normalised_frames.transpose()]
        dataset=(pd.concat(final_frames)).transpose()
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        return dataset
    def store_normalised_data(self,df, function):
        dataset = function(df.iloc[:, :-1].replace([np.inf, -np.inf], np.nan).fillna(0))
        final_dataset = [dataset, df.iloc[:, -1]]#.transpose()
        dataset = (pd.concat(final_dataset, axis = 1) )#.transpose()
        dataset = dataset.replace([np.inf, -np.inf], np.nan).fillna(0)
        return dataset
    def make_RF_normalization(self,df):
        feat_labels = []
        num_of_features = -1
        for feat in df:
            feat_labels.append(feat)
            num_of_features +=1
        X = df.iloc[:, :num_of_features] # select all columns containing features, column 0 is the Date
        y = df.iloc[:, num_of_features]
        # Split the data into 20% test and 80% training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False, random_state=0)
        # Noise
        mu, sigma = 0, 0.1
        # Create a random forest classifier
        num_of_trees = 100
        clf = RandomForestRegressor(n_estimators=num_of_trees, random_state=0, n_jobs=-1, min_samples_leaf=5,
                                         oob_score=True)
        rf = clone(clf)
        rf2 = clone(clf)
        # Train the classifier
        rf.fit(X_train, y_train)
        #Save the oob score
        oob_score_before_noise = rf.oob_score_
        oobv_scores = []
        for feat in feat_labels[:-1]:
            # Copy the original X_train
            X_train_temp = X_train.copy()
            # Copy the original column of the current feature
            temp_feat_column = X_train[feat].copy()
            # Randomly change the value of SOME (30%) of the data -> size=int(0.3*(len(X_train))),
            # but int(len(X_train)) for the selected indices to be randomly distributed
            indices_to_add_noise = np.random.randint(int(len(X_train)), size=int(0.3 * (len(X_train))))
            for i in indices_to_add_noise:
                temp_feat_column[i] += np.random.normal(mu, sigma)
            # Remove original feature column
            X_train_temp.drop(feat, axis="columns", inplace=True)
            # Add feature column with noise
            X_train_temp[feat] = temp_feat_column
            rf2.fit(X_train_temp, y_train)
            oobv_scores.append(oob_score_before_noise - rf2.oob_score_)
        feat_df = pd.DataFrame(list(zip(feat_labels, oobv_scores)),
                       columns =['Features', 'oobv'])
        feat_df = feat_df.sort_values(by=['oobv'], ascending=False)
        feat_list = feat_df.Features.values[:round(0.8*len(feat_df.Features.values))].tolist()
        feat_list.append(feat_labels[-1])
        return df[feat_list]


    def estimate_final_indices(self, asset, start, end):
        dbManager = DBManager()
        indices = dbManager.get_asset_indices(asset['asset_id'], start, end)['asset_indices']
        df = pd.DataFrame(indices)
        df = df.rename(columns={"close": "Close"})
        df = df.rename(columns={"Low": "low", "High": "high", "Open": "open", "Volume": "volume"})
        df['ao_oscillator'] = self.ao_oscillator(df)
        df['Disp5'] = self.Disp_n(df, 5)
        df['Disp10'] = self.Disp_n(df, 10)
        df['MA5'] = self.moving_average(df, 5)
        df['OSCP'] = self.OSCP(df, 5, 10)
        df['BIAS_6'] = self.BIAS(df, 6)
        df['ASY1'] = self.ASY_1(df)
        df['ASY2'] = self.ASY(df, 2)
        df['ASY3'] = self.ASY(df, 3)
        df['ASY4'] = self.ASY(df, 4)
        df['ASY5'] = self.ASY(df, 5)
        df['Roll_Var'] = self.moving_variance(df, 21)
        df['MA_26'] = self.moving_average(df, 26)
        df['MA_20'] = self.moving_average(df, 20)
        df['VMACD1'] = self.VMACD(df, 20, 40, 9, True).iloc[:, 0]
        df['VMACDsign1'] = self.VMACD(df, 20, 40, 9, True).iloc[:, 1]
        df['VMACDdiff1'] = self.VMACD(df, 20, 40, 9, True).iloc[:, 0] - self.VMACD(df, 20, 40, 9, True).iloc[:, 1]
        df['vma'] = self.vol_moving_average(df, 3)
        df = df.rename(columns={"Close": "close", "Low": "low", "High": "high", "Open": "open", "Volume": "volume"})
        MACD = TA.MACD(df, 40, 20)
        df['MACD'] = MACD.iloc[:, 0]
        df['MACDsign'] = MACD.iloc[:, 1]
        df['MACDdiff'] = MACD.iloc[:, 0] - MACD.iloc[:, 1]
        function_list = [TA.ROC, TA.STOCH, TA.MOM, TA.WILLIAMS, TA.EMA, TA.CCI, TA.RSI, TA.ATR, TA.TRIX, TA.VAMA, TA.SAR, TA.VPT]
        attribute_list = ['ROC', 'K', 'MOM', '%R Williams', 'EXPM', 'CCI', 'RSI', 'ATR', 'TRIX', 'VAMA', 'SAR', 'VPT']
        function_list_comb = [ TA.PPO]
        # Estimate and concatenate to pandas dataframe features that included in function_list and attribute_list
        for k in range(0, len(function_list)):
                df['{}'.format(attribute_list[k])] = self.estimate_Ta_attributes(attribute_list[k], df, function_list[k])
            # Estimate and concatenate to pandas dataframe features that included in function_list_comb and attribute_list_comb
        for l in range(0, len(function_list_comb)):
                feature = self.estimate_Ta_comb_attributes(df, function_list_comb[l])
                for m in range(0, len(feature.iloc[1, :])):
                    df['{}'.format(feature.columns[m])] = feature.iloc[:, m]
        pyti_feat_list_one_arg = [ VOLAT, DPO, STDDV]
        pyti_attr_list_one_arg = [ 'VOLAT', 'DPO', 'STDDV']
        for n in range(0, len(pyti_attr_list_one_arg)):
                df['{}'.format(pyti_attr_list_one_arg[n])] = self.estimate_pyti_attributes_one_arg(pyti_feat_list_one_arg[n], df, 5)
                df['{}'.format(pyti_attr_list_one_arg[n])] = df['{}'.format(pyti_attr_list_one_arg[n])]
        pyti_feat_list_two_arg = [POSC, VO]
        pyti_attr_list_two_arg = ['POSC', 'VO']
        for p in range(0, len(pyti_attr_list_two_arg)):
                df['{}'.format(pyti_attr_list_two_arg[p])] = self.estimate_pyti_attributes_two_arg(pyti_feat_list_two_arg[p], df, 5, 25)
                df['{}'.format(pyti_attr_list_two_arg[p])] = df['{}'.format(pyti_attr_list_two_arg[p])]
        df['VSTDDV'] = self.standard_deviation(df.volume, 5)
        df.reset_index(inplace=True)
        df['PSY_12'] = self.PSY(df, 12)
        df['Roll_Var_Rat'] = self.moving_variance_ratio(df)
        df['ASMR'] = self.ASY(df, 21)
        del df['high']
        del df['open']
        del df['volume']
        del df['low']
        cols = list(df.columns)
        c, d = cols.index('close'), cols.index('ASMR')
        cols[d], cols[c] = cols[c], cols[d]
        df = df[cols]
        columns = df.columns[1:]
        df = pd.DataFrame(KNN(k=5).fit_transform(df.iloc[:, 1:]), columns=columns)
        data = self.store_normalised_data(df, self.minmax_scale)
        close_data = data.iloc[:, -1].to_frame()
        frames = []
        for x in close_data:
                transformer = MinMaxScaler().fit(close_data[x].values.reshape(-1, 1))
                close_data[x] = transformer.transform(close_data[x].values.reshape(-1, 1))
                frames.append(close_data[x])
        data = [data.iloc[:, :-1], close_data]
        final_data = pd.concat(data, axis=1)

        output_dir = os.path.join(signal_dir, "signal_models", asset['name'])
        output_file_dir = os.path.join(output_dir, 'my_features.txt')
        with open(output_file_dir, 'a') as file:
                file.write(str(final_data.columns) + '\n')
        output_file_dir = os.path.join(output_dir, asset['name'] + '.csv') 
        final_data.to_csv(output_file_dir)
        return final_data
