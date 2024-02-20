from datetime import datetime, timedelta, timezone, date
import time
import json
import pandas as pd

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class GeneralUtils(metaclass=Singleton):

    current_timestamp = -1

    def setCurrentTimestsamp(self, value):
        self.current_timestamp = value

    def getCurrentTimestsamp(self):
        self.current_timestamp = int(round(time.time() * 1000))
        return self.current_timestamp

    def convertDaysToMiliseconds(self, days):
        return days*86400000
    
    def convertDateToMilliSecs(self, date):
        # e.g. date '08/09/2020 00:00:00'
        dt_obj = datetime.strptime(date, '%d/%m/%Y %H:%M:%S')
        millisec = dt_obj.timestamp() * 1000
        return millisec
    
    def convertMillisecsToDate(self, millis):
        return time.strftime('%Y-%m-%d', time.gmtime(millis/1000.0))

    def getStartEndFromDate(self, date):
        dt_obj = datetime.utcfromtimestamp(date/1000.0)
        start = int(dt_obj.replace(minute=0, hour=0, second=0, microsecond=0, tzinfo=timezone.utc).timestamp() * 1000)
        end = int(dt_obj.replace(minute=59, hour=23, second=59, microsecond=0, tzinfo=timezone.utc).timestamp() * 1000)
        return start, end
    
    def getTodayStartEndTimestamps(self):        
        today_timestamp = self.getCurrentTimestsamp()
        date_start, date_end = self.getStartEndFromDate(today_timestamp)
        return date_start, date_end

    def getCurrentDate(self):
        timestamp = self.getCurrentTimestsamp() / 1000
        dt_object = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        current_data = dt_object.strftime("%Y-%m-%d")
        return current_data

    def sql_to_json(self, data, cursor):
        try:
            json_data=[]
            row_headers=[x[0] for x in cursor.description]
            for result in data:
                json_data.append(dict(zip(row_headers,result)))
        except Exception as e:
            json_data=[]
            print("====================" + str(e) + "====================")
        finally:
            return json_data

    def sql_to_dataframe(self, data, cursor, iloc_enabled=False):
        try:
            field_names = [i[0] for i in cursor.description]
            df = pd.DataFrame(data, columns=field_names)
            if iloc_enabled:
                df = df.iloc[0]
        except Exception as e:
            df = None
            print("====================" + str(e) + "====================")
        finally:
            return df

    def remove_emoji(self, inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')




