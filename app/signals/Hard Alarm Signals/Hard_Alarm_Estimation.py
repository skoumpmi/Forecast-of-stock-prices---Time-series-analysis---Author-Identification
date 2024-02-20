
class Hard_Alarm():
    def emergency_data(self, df):
        data = df.copy().reset_index()
        today_data = (data.iloc[len(data.index)-1]).to_frame().transpose().reset_index(drop=True)
        before_data = (data.iloc[:len(data.index)-1]).reset_index(drop=True)
        max = before_data['High'].max()
        min = before_data['Low'].min()
        today_data['Emerg_High'] = 0
        today_data['Emerg_Low'] = 0
        for i in range(0, len(today_data)):
            if today_data.iloc[i]['High'] > max:
                today_data.loc[i, 'Emerg_High'] = 1
            if today_data.iloc[i]['Low'] < min:
                data.loc[i, 'Emerg_Low'] = 1
        for i in range(0, len(data) - 1):
            data.loc[i + 1, 'Return'] = (data.loc[i + 1, 'Close'] - data.loc[i, 'Close']) / data.loc[i, 'Close']
        std_return = data['Return'].std()
        today_return = data['Return'].to_frame().iloc[-1].values[0]
        today_data['Return'] = today_return
        today_data['Emer_Return'] = 0
        Emerg_list = []
        for i in range(0, len(today_data)):
            if abs(today_data.iloc[i]['Return']) > abs(2*std_return):
                Emerg_list.append(1)
            else:
                Emerg_list.append(0)
        today_data['Emer_Return'] = Emerg_list
        Return_Volume_list = [0.0]
        for z in range(0, len(data) - 1):
            Return_Volume_list.append((data.iloc[z + 1]['Volume'] - data.iloc[z]['Volume']) / data.iloc[z]['Volume'])
        data['Return_Volume'] = Return_Volume_list
        std_volume = data['Return_Volume'].std()
        today_return_volume = data['Return_Volume'].to_frame().iloc[-1].values[0]
        today_data['Return_Volume'] = today_return_volume
        today_data['Emer_Volume'] = 0
        Emerg_volume_list = []
        for i in range(0, len(today_data)):
            if abs(today_data.iloc[i]['Return_Volume']) > abs(2 * std_volume):
                Emerg_volume_list.append(1)
            else:
                Emerg_volume_list.append(0)
        today_data['Emer_Volume'] = Emerg_volume_list

        mean = data['Close'].mean()
        data['Emer_Close'] = 0.0
        Var_Close_list = []
        for i in range(0, len(data)):
            Var_Close_list.append(pow((data.iloc[i]['Close'] - mean), 2))
        data['Var_Close'] = Var_Close_list
        today_var_close = data['Var_Close'].to_frame().iloc[-1].values[0]
        today_data['Var_Close'] = today_var_close
        Emerg_close_list = []
        for i in range(0, len(today_data)):
            if today_data.iloc[i]['Var_Close'] > data['Var_Close'].quantile(.99):
                Emerg_close_list.append(1)
            else:
                Emerg_close_list.append(0)
        today_data['Emer_Close'] = Emerg_close_list
        today_data.reset_index(inplace=True)
        today_data["Hard_Alarm"] = 0.0
        for i in range(0, len(today_data)):
            if today_data['Emer_Close'][i] == 1 or today_data['Emerg_Low'][i] == 1 or today_data['Emerg_High'][i] == 1 \
                    or today_data['Emer_Return'][i] == 1 or today_data['Emer_Volume'][i] == 1:
                today_data["Hard_Alarm"][i] = 2.0
        today_data = today_data['Hard_Alarm'].to_frame()
        return today_data.iloc[-1, -1]


