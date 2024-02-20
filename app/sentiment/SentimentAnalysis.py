import numpy as np
import pandas as pd
import datetime
import os
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

import sys
sys.path.insert(1, './BERT')

from .BERT.bert_classifiers import linearClassifier
from .BERT.tools import InputExample, BinaryClassificationProcessor
from .BERT.convert_examples_to_features import convert_example_to_feature

from transformers import BertTokenizer
from multiprocessing import Pool, cpu_count
from contextlib import closing

from tqdm import tqdm

from scipy.special import softmax
from scipy.stats import norm, t

from ..database.dbManager import DBManager
from ..utilities.generalUtils import GeneralUtils


# decimal points to round everything to
decimal_points = 3

class SentimentAnalysis:      
    def __init__(self, 
                decay_past_window = 90, 
                weight = 0.035,
                sentiment_change_window = 90, 
                ROC_past_window = 10, 
                threshold = 1.1,
                buy_signal = 1,
                sell_signal = 2,
                min_signal_probability = 0.55,
                max_signal_probability = 0.75,
                signal_type = 2):
        
        self.path_to_model_directory = os.path.join(os.getcwd(), "app", "sentiment", "finetuned_BERT_model_files")
        self.model, self.tokenizer = self.load_model_and_tokenizer(self.path_to_model_directory)
        self.db_manager = DBManager()
        self.gu = GeneralUtils()
        self.decay_past_window = decay_past_window
        self.weight = weight
        self.sentiment_change_window = sentiment_change_window 
        self.ROC_past_window = ROC_past_window
        self.threshold = threshold
        self.buy_signal = buy_signal
        self.sell_signal = sell_signal
        self.min_signal_probability = min_signal_probability
        self.max_signal_probability = max_signal_probability
        self.signal_type = signal_type

    
    def load_model_and_tokenizer(self, path_to_model_directory):
        """ "path_to_directory" is the directory where the model elements
        (pytorch_model.bin, config.json and vocab.txt) are be stored """
        tokenizer = BertTokenizer.from_pretrained(os.path.join(path_to_model_directory), do_lower_case=False)
        finetuned_model = linearClassifier.from_pretrained(path_to_model_directory)  
        model = finetuned_model.eval()
        
        return model, tokenizer
    

    def generate_eval_features(self, eval_examples_for_processing, eval_examples_len):
    
        process_count = cpu_count() - 1 
        eval_features = list(tqdm(map(convert_example_to_feature, \
                                            eval_examples_for_processing), total=eval_examples_len))
           
        return eval_features
    
    
    def predict_sentiment(self, text):
        """ predicts the sentiment of the text using the finetuned_model, after
            tokenizing the text with the given tokenizer """
        """ select gpu or cpu"""   
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = "cpu"        
        max_seq_length =  128 # or 512, that's the number of possible tokens not words
        eval_batch_size = 8      
        processor = BinaryClassificationProcessor()      
        eval_examples = [InputExample(guid=0, text_a=text, text_b=None, label='1')]       
        label_list = ["0", "1", "2", "3", "4"]
        eval_examples_len = len(eval_examples)        
        label_map = {label: i for i, label in enumerate(label_list)}
        eval_examples_for_processing = [(example, label_map, max_seq_length, self.tokenizer, "classification") \
                                        for example in eval_examples]
        
        eval_features =  self.generate_eval_features(eval_examples_for_processing, eval_examples_len)                     
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        # since "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)  
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)        
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size = eval_batch_size)      
        preds = []
        
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
        
            with torch.no_grad():
                output = self.model(input_ids, segment_ids, input_mask, labels=None)
                logits = output[0]
            
            if len(preds)==0:
                preds.append(logits.cpu().numpy())
            else: 
                preds[0] = np.append(preds[0], logits.cpu().numpy(), axis = 0)
                
        preds = preds[0]
        pred_probs = softmax(preds, axis=1)
        predicted_class_probability = np.max(pred_probs)        
        predicted_class = np.argmax(pred_probs)        
        predicted_class = predicted_class - 2 # mapping [0,1,2,3,4] to [-2, -1, 0, 1, 2]
        predicted_continuous_sentiment = np.sum(pred_probs*np.array([-2, -1, 0, 1, 2]))

        return predicted_class, predicted_class_probability, predicted_continuous_sentiment


    def compute_confidence_level(self, reports_sents, decayed_sents):
        sample = reports_sents  
        sample_size = len(sample)
        sample_std = np.std(sample)
        sample_mean = np.mean(sample)
        # compute degrees of freedom -> doof
        doof = sample_size - 1     

        if (reports_sents and decayed_sents):
            # in case we have one report only 
            # or a (small) number of reports that happen
            # to have the same exact sentiment
            population_approximation = decayed_sents # + reports_sents
            population_mean = np.mean(population_approximation)
            if sample_std == 0:
                population_std = np.std(population_approximation)
                if (population_std!=0):
                    z_score = (sample_mean - population_mean)/population_std
                    confidence_level = 1 - norm.cdf(z_score)/2
                else:
                    # rare case, not enough information ->
                    # no means to infer a confidence level,
                    # confidence level is assigned a low enough value 
                    confidence_level = 0.5 
            else:  
                t_statistic = (sample_mean - population_mean)/(sample_std/np.sqrt(sample_size))
                t_statistic_abs = abs(t_statistic)                                           
                # confidence level is given by the stats.t.cdf function
                confidence_level = t.cdf(t_statistic_abs, doof)
        elif (reports_sents and not(decayed_sents)):
            # in case we have no decayed sentiments
            if sample_size == 1:
                # in case we have a one report only, not enough information ->
                # no means to infer a confidence level,
                # confidence level is assigned a low enough value 
                confidence_level = 0.5 
            elif sample_std == 0 :
                # rare case
                confidence_level = 0.75
            else:
                # sample_mean -> SAMPLE MEAN
                # round(sample_mean) -> approximation for POPULATION MEAN
                t_statistic = (sample_mean - round(sample_mean))/(sample_std/np.sqrt(sample_size))
                t_statistic_abs = abs(t_statistic)                                           
                confidence_level = t.cdf(t_statistic_abs, doof)     
        else:
            print("This part of code should have not been reached.")
            confidence_level = None 
            
        return confidence_level


    def compute_total_sentiment_probability(self,
                                selected_version,
                                sentiments_classes,
                                sentiments_probabilities,
                                continuous_sentiments,
                                decayed_sentiments,
                                decayed_sentiments_probabilities):
        
        if (selected_version == "sentiments_classes"):
            probability = self.compute_confidence_level(sentiments_classes, 
                                                        decayed_sentiments)
        elif (selected_version == "continuous_sentiments"):
            probability = self.compute_confidence_level(continuous_sentiments, 
                                                        decayed_sentiments)
        elif (selected_version == "sentiments_probabilities_products"):
            reports_sentiments_probabilities_prods = \
                sentiments_classes*sentiments_probabilities
            decayed_sentiments_probabilities_prods = \
                decayed_sentiments*decayed_sentiments_probabilities                
            probability = self.compute_confidence_level(reports_sentiments_probabilities_prods, 
                                                        decayed_sentiments_probabilities_prods)

        return probability    


    def squeeze_probability(self, probability):
        """ squeeze probability value to [min_signal_probability, max_signal_probability] interval"""
        squeezed_probability = self.min_signal_probability + \
             (self.max_signal_probability-self.min_signal_probability)*probability
        return squeezed_probability


    def get_reports_sentiments(self, asset_id, reports_df, model, tokenizer):
        """ returns the mean sentiment of the new reports concerning the asset with asset_id"""       
        
        # stores the sentiments of all the reports for the asset_id
        sentiments_classes = []
        probabilities = []
        continuous_sentiments = []
        reports_for_asset_df = reports_df.loc[(reports_df['asset_id'] == asset_id)]       
        
        # if there are any new reports for the current asset_id
        if (len(reports_for_asset_df)!=0):
            # predict the sentiment for each of them
            for report_num in range(len(reports_for_asset_df)):            
                report_id = reports_for_asset_df.iloc[report_num]['report_id']
                print(report_id)
                # get the text of the report
                report_text = reports_for_asset_df.iloc[report_num]['full_text']
                sentiment_class, probability, continuous_sentiment = \
                    self.predict_sentiment(report_text)                                       
                sentiments_classes.append(sentiment_class)
                probabilities.append(probability)
                continuous_sentiments.append(continuous_sentiment)
                self.db_manager.update_report_sentiment_in_db(report_id, sentiment_class)

        return sentiments_classes, probabilities, continuous_sentiments
    
    
    def compute_decay_single_day(self,
                                initial_sentiment_datetime_timestamp, 
                                initial_sentiment_value, 
                                current_datetime_timestamp):
        """Returns the decayed value (for the "current date"") of the "initial sentiment" 
        that began on "initial_sentiment_datetime_timestamp"-date """ 
        
        weights_length = self.decay_past_window
        weight = self.weight                  
        # the beginning and end of the current date in miliseconds
        current_date_start_timestamp, current_date_end_timestamp = \
            self.gu.getStartEndFromDate(current_datetime_timestamp)
        # current_date to be a datetime.date object 
        current_date = self.gu.convertMillisecsToDate(current_date_start_timestamp)           
        # get the begin_date for the current sentiment 
        begin_date_start_timestamp, begin_date_end_timestamp = \
            self.gu.getStartEndFromDate(initial_sentiment_datetime_timestamp)         
        begin_date = self.gu.convertMillisecsToDate(begin_date_start_timestamp)     
        
        # in order to compute the decay only up to the current date
        current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d").date()
        begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d").date()
        timedelta_of_dates = current_date - begin_date 
        timedelta_of_days = timedelta_of_dates.days
        
        if (timedelta_of_days == 0):
            decayed_sentiment_for_current_date = initial_sentiment_value
        elif (timedelta_of_days != 0 and timedelta_of_days <= weights_length):
            weights_length = timedelta_of_days          
            # set end date
            end_date = begin_date + timedelta(days=weights_length)        
            weightspace = np.exp(np.linspace(weights_length, 0, num=weights_length+1)*-weight)[::-1]    
            d = {'date': [begin_date, end_date], 'sentiment_value' :[initial_sentiment_value, 0]}   
            df = pd.DataFrame(data=d)
            df['date']= pd.to_datetime(df['date'])
            df.set_index('date', inplace = True)
            # resample('1D') -> sentiment decays during all the days of the week
            # if resample('1B') -> sentiment decayd only during the business days of the week            
            df = df.resample('1D').bfill().reset_index()
            df['dec_sent'] = weightspace * initial_sentiment_value        
            df['weights'] = weightspace
            decayed_sentiment_for_current_date = df["dec_sent"].iloc[-1]    
        else:
            decayed_sentiment_for_current_date = 0
        
        return decayed_sentiment_for_current_date
    
    
    def get_decayed_sentiments(self, 
                                asset_id, 
                                current_datetime_timestamp, 
                                sentiment_history_df):
        """ Computes the average of the decayed sentiments of previous days for the
        current date"""
        decayed_sentiments = []
        decayed_sentiments_probabilities = []
        
        # the beginning and end of the current date in miliseconds
        start, end = self.gu.getStartEndFromDate(current_datetime_timestamp)

        # the initial date
        from_datetime_timestamp = current_datetime_timestamp - self.gu.convertDaysToMiliseconds(self.decay_past_window)

        # select rows that have information for the current asset_id
        mask = (sentiment_history_df['asset_id']==asset_id) & \
            (sentiment_history_df['date'] > from_datetime_timestamp) & \
            (sentiment_history_df['date'] < end) 

        current_asset_history_df = sentiment_history_df[mask]             
        # previous dates & sentiments are binded using zip
        previous_dates = current_asset_history_df['date'].tolist()
        previous_sentiments = current_asset_history_df['sentiment'].tolist()
        previous_probabilities = current_asset_history_df['probability'].tolist()
        
        if (len(previous_dates)!=0):
            for initial_sentiment_datetime_timestamp, \
                initial_sentiment_value, \
                initial_sentiment_probability \
                in zip(previous_dates, previous_sentiments, previous_probabilities):
                # elegxos na diaforetiko apo to miden
                decayed_sentiment_value = \
                    self.compute_decay_single_day(initial_sentiment_datetime_timestamp, 
                                                    initial_sentiment_value, 
                                                    current_datetime_timestamp)
                decayed_sentiments.append(decayed_sentiment_value)
                decayed_sentiments_probabilities.append(initial_sentiment_probability)
    
        print("The decayed sentiments are: ", decayed_sentiments)
        print("The decayed probabilities are: ", decayed_sentiments_probabilities)

        return decayed_sentiments, decayed_sentiments_probabilities
    

    def check_sentiment_change(self, 
                                asset_id, 
                                current_datetime_timestamp, 
                                total_sentiment, 
                                sentiment_change_df):
        
        current_sentiment = total_sentiment
        # the beginning and end of the current date in miliseconds
        start, end = self.gu.getStartEndFromDate(current_datetime_timestamp)
        sentiment_change_window = self.sentiment_change_window
        # subtract one day ->(convertDaysToMiliseconds(1)) 
        # to get the start & end milliseconds values of the previous day        
        previous_date_milliseconds_start, previous_date_milliseconds_end = \
            self.gu.getStartEndFromDate(start - \
                self.gu.convertDaysToMiliseconds(sentiment_change_window)) 
        
        # previous_date_milliseconds_start =  start timestamp of the day - sentiment_change_window 
        mask = (sentiment_change_df['asset_id']==asset_id) & \
                (sentiment_change_df['date'] >= previous_date_milliseconds_start) & \
                (sentiment_change_df['date'] < current_datetime_timestamp)
                
        previous_sentiments_df = sentiment_change_df[mask]
        if (len(previous_sentiments_df)==0):
            print("========= No previous sentiment recorded for asset_id", asset_id, "=========")
            # in case no previous sentiment exist, in order to create a first record:
            previous_sentiment = current_sentiment + 1 
            self.db_manager.insert_sentiment_change(previous_sentiment, 
                                                    int(current_sentiment), 
                                                    current_datetime_timestamp, 
                                                    asset_id)
        else:             
            # previous_sentiment -> the most recent recorded sentiment
            previous_sentiments_df = previous_sentiments_df.sort_values(by='date', ascending=True)
            previous_sentiment = previous_sentiments_df.iloc[-1]['current_sentiment']
            if (int(current_sentiment)!= int(previous_sentiment)):
                self.db_manager.insert_sentiment_change(int(previous_sentiment), 
                                                        int(current_sentiment), 
                                                        current_datetime_timestamp, 
                                                        asset_id)
        

    """ Computes total sentiment (latest reports sentiment + decayed sentiment) """
    def compute_total_sentiment(self, 
                                asset_id, 
                                reports_df, 
                                current_datetime_timestamp, 
                                sentiment_history_df):
  
        # reports' sentiments
        sentiments_classes, sentiments_probabilities, continuous_sentiments = \
            self.get_reports_sentiments(asset_id, reports_df, self.model, self.tokenizer)

        # decayed sentiments
        decayed_sentiments, decayed_sentiments_probabilities = \
            self.get_decayed_sentiments(asset_id, current_datetime_timestamp, sentiment_history_df)
        
        # remove zero elements from decayed sentiments
        try:
            non_zero_decayed_sentiments, non_zero_decayed_sentiments_probabilities = \
                    zip(*((x, y) for x, y in zip(decayed_sentiments, decayed_sentiments_probabilities) if x!=0))
        except:
            non_zero_decayed_sentiments = []
            non_zero_decayed_sentiments_probabilities = []

        ''' 3 versions of inputs for the computation of probability:
         - "sentiments_classes"
         - "continuous_sentiments"
         - "sentiments_probabilities_products"
         '''    
        probability_version = "continuous_sentiments"    

        if (not(sentiments_classes) and not(non_zero_decayed_sentiments)):
            sentiment_from_reports = 0
            decayed_sentiment = 0
            total_sentiment = 0
            total_sentiment_probability = 0
        else:
            if (not(sentiments_classes) and non_zero_decayed_sentiments):
                sentiment_from_reports = 0 # 
                decayed_sentiment = np.mean(non_zero_decayed_sentiments)
                total_sentiment = decayed_sentiment
                # current date's probability = previous date's probability,
                # (as we had no new reports) 
                total_sentiment_probability = self.db_manager.get_last_probability_for_asset(asset_id)
            elif (sentiments_classes and not(non_zero_decayed_sentiments)):
                sentiment_from_reports = np.mean(sentiments_classes)
                decayed_sentiment = 0
                total_sentiment = np.mean(sentiments_classes)
                total_sentiment_probability = self.compute_total_sentiment_probability(probability_version,
                                                                                        sentiments_classes,
                                                                                        sentiments_probabilities,
                                                                                        continuous_sentiments,
                                                                                        non_zero_decayed_sentiments,
                                                                                        non_zero_decayed_sentiments_probabilities)
            else:
                # both mean_reports_sentiment & mean_decayed_sentiment contribut
                # equally to the current total sentiment
                sentiment_from_reports = np.mean(sentiments_classes)
                decayed_sentiment = np.mean(non_zero_decayed_sentiments)
                total_sentiment = (sentiment_from_reports + decayed_sentiment)/2
                total_sentiment_probability = self.compute_total_sentiment_probability(probability_version,
                                                                                        sentiments_classes,
                                                                                        sentiments_probabilities,
                                                                                        continuous_sentiments,
                                                                                        non_zero_decayed_sentiments,
                                                                                        non_zero_decayed_sentiments_probabilities)


        print("============= Asset ID: ", asset_id, "==============")
        print("Sentiment from reports is: ", sentiment_from_reports)
        print("Decayed sentiment is: ", decayed_sentiment)
        print("Asset sentiment is: ", total_sentiment)
        print("Asset sentiment probability is: ", total_sentiment_probability)
        print("==================================================")

        return float(total_sentiment), float(total_sentiment_probability)


    def get_sentiment(self, asset_id, asset_sentiment_df):
           
        start_timestamp, end_timestamp = self.gu.getTodayStartEndTimestamps()           
        # select rows that correspond to asset_id and the timestamps that belong to the current day
        mask = (asset_sentiment_df['asset_id'] == asset_id) & \
                (asset_sentiment_df['date']  >= start_timestamp) & \
                (asset_sentiment_df['date']  <= end_timestamp)
       
        todays_sentiments_df = asset_sentiment_df[mask] 
        """ in case we have more than one sentiments recorded for the current date: """  
        last_sentiment = float(todays_sentiments_df.iloc[-1]['sentiment'])

        return last_sentiment

################################################################################################################################

    def calculate_sentiment_for_all_assets(self):        
        current_datetime_timestamp = self.gu.getCurrentTimestsamp()        
        from_datetime_timestamp = current_datetime_timestamp - \
                    self.gu.convertDaysToMiliseconds(self.decay_past_window)
                            
        reports_df  = self.db_manager.get_unprocessed_reports()      
        sentiment_history_df = self.db_manager.get_assets_sentiment(from_datetime_timestamp, 
                                                                    current_datetime_timestamp)      
        sentiment_change_df = self.db_manager.get_sentiment_change_for_backend(from_datetime_timestamp, 
                                                                                current_datetime_timestamp)        
         
        all_assets = self.db_manager.get_all_assets()
        for asset in all_assets: 
            sentiment, probability = \
                self.compute_total_sentiment(asset["asset_id"], 
                                            reports_df, 
                                            current_datetime_timestamp, 
                                            sentiment_history_df)
            if (sentiment == 0 and probability == 0):
                pass
                # No insertion into the database happens 
            else:
                sentiment = int(round(sentiment))
                probability = float(round(probability, decimal_points))   
                self.db_manager.insert_asset_sentiment(asset["asset_id"], 
                                                        sentiment, 
                                                        probability, 
                                                        current_datetime_timestamp)            
                self.check_sentiment_change(asset["asset_id"], 
                                            current_datetime_timestamp, 
                                            sentiment, 
                                            sentiment_change_df)

################################################################################################################################

    def generate_sentiment_signal_for_all_assets(self):     
        signal_type = self.signal_type
        assets = self.db_manager.get_all_assets()
        for asset in assets:
            asset_sentiment_signal, signal_probability = self.generate_sentiment_signal(asset["asset_id"])
            if asset_sentiment_signal!=0:
                current_timestamp = self.gu.getCurrentTimestsamp()
                self.db_manager.insert_signal(asset["asset_id"], signal_probability, asset_sentiment_signal, signal_type, current_timestamp)  

    def generate_sentiment_signal(self, asset_id):
        signal = 0
        try:      
            sentiment = self.db_manager.get_last_sentiment_for_asset(asset_id)
            probability = self.db_manager.get_last_probability_for_asset(asset_id)
            signal_probability = self.squeeze_probability(probability)
            indeces_window = self.db_manager.get_indeces_with_window_by_asset(asset_id, 
                                                                            self.ROC_past_window )
            ROC = (float(indeces_window['last']) - float(indeces_window['first']))/float(indeces_window['first'])
            if ROC==0:
                print("There is no ROC value available for asset: " + str(asset_id))    
            elif (abs(sentiment)> self.threshold and (np.sign(sentiment)==np.sign(ROC))):
                if (np.sign(sentiment)==1):
                    signal = self.buy_signal
                elif (np.sign(sentiment)==-1):
                    signal = self.sell_signal
        except Exception as e:
                print(str(e))
        finally:
            return signal, signal_probability