import tweepy
import requests
import configparser
from datetime import datetime
import calendar
import json
import joblib
import os
import pandas as pd
import collections
from ..database.dbManager import DBManager
from ..utilities.generalUtils import GeneralUtils

config = configparser.ConfigParser()
config.read('config.ini')

class TwitterStreamer:

    def __init__(self):
        auth = tweepy.OAuthHandler(config['twitter']['consumer_key'], config['twitter']['consumer_secret'])
        auth.set_access_token(config['twitter']['access_token'], config['twitter']['access_token_secret'])
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.db_manager = DBManager()
        self.utils = GeneralUtils()

    
    def get_tweets_for_all_assets(self):
        success = False
        try:
            print("-------- Get New Tweets --------")
            assets = self.db_manager.get_all_assets()
            
            for asset in assets:
                print("Asset Name: " + str(asset["name"]))
                tweets_result = self.stream_tweets_by_query(asset, 'twitter')
                print("Tweets Retrieved: " + str(len(tweets_result)))
                if len(tweets_result) > 0:
                    success_save = self.db_manager.insert_tweets(tweets_result)
                    print("Stored: " + str(success_save))
                print("-------------------------------")
            success = True
        except Exception as e:
            success = False
            print("-------- PROBLEM: New Tweets NOT updated--------")  
            print("====================" + str(e) + "====================")
        finally:
            return success

    def get_stocktwits_for_all_assets(self):
        success = False
        try:
            print("-------- Get New StockTwits --------")
            assets = self.db_manager.get_all_assets()
            for asset in assets:
                print("Asset Name: " + str(asset["name"]))
                stocktwits_result = self.stream_stocktwits_by_query(asset, "stocktwits")
                print("Stocktwits Retrieved: " + str(len(stocktwits_result)))
                if len(stocktwits_result)>0:
                    success_save = self.db_manager.insert_tweets(stocktwits_result)
                    print("Stored: " + str(success_save))
                print("-------------------------------")
            success = True
        except Exception as e:
            success = False
            print("-------- PROBLEM: New StockTwits  NOT updated--------")
            print("====================" + str(e) + "====================")
        finally:
            return success

    def stream_tweets_by_query(self, asset, platform, lang="en"):
        count = 100
        new_tweet_list = []
        tweets_list = []
        try:
            query_list = self.db_manager.get_asset_twitter_queries(asset["asset_id"])
            query = ' OR '.join(query_list)
            for tweet in tweepy.Cursor(self.api.search, q=query, lang=lang).items(count):
                metadata = {
                    "user_verified": tweet.user.verified, 
                    "user_created_at": calendar.timegm(tweet.user.created_at.timetuple())*1000, # tweet.user.created_at.strftime("%d/%m/%Y %H:%M"),
                    "followers": tweet.user.followers_count, 
                    "status_count":  tweet.user.statuses_count,
                    "friends": tweet.user.friends_count, 
                    "mentionCount": len(tweet.entities["user_mentions"]),
                    "urlCount": len(tweet.entities["urls"]), 
                    "hashtagCount": len(tweet.entities["hashtags"]),
                    "retweetCount":  tweet.retweet_count
                    }

                formatted_tweet = {
                    "tweet_id": int(tweet.id), 
                    "text": self.utils.remove_emoji(tweet.text), 
                    "user": tweet.user.screen_name,
                    "create_date": calendar.timegm(tweet.created_at.timetuple())*1000,
                    "retrieve_date": self.utils.getCurrentTimestsamp(),
                    "sentiment": None, 
                    "platform": 'twitter', 
                    "asset_id": asset["asset_id"], 
                    "metadata": metadata
                    }
                is_reliable = self.get_tweet_reliability(metadata)
                if is_reliable == 1:
                    tweets_list.append(formatted_tweet)
            
            if len(tweets_list) > 0:
                tweets_list.reverse()
                new_tweet_list = self.db_manager.remove_duplicate_tweets(tweets_list, platform)

        except Exception as e:
            print("====================" + str(e) + "====================")
        return new_tweet_list

    # get the data using StockTwits API
    def stream_stocktwits_by_query(self, asset, platform):
        new_tweet_list = []
        tweets_list = []
        query = self.db_manager.get_asset_stocktwits_query(asset["asset_id"])
        url = "https://api.stocktwits.com/api/2/streams/symbol/{0}.json".format(query)
        try:
            response = requests.get(url).json()
            if "messages" in response:
                for tweet in response['messages']:
                    metadata = {
                        "user_verified": None,
                        "user_created_at": calendar.timegm(datetime.strptime(tweet['user']['join_date'], '%Y-%m-%d').timetuple())*1000,
                        "followers": tweet["user"]["followers"],
                        "following": tweet["user"]["following"],
                        "status_count": None,
                        "friends": None,
                        "mentionCount": len(tweet["mentioned_users"]),
                        "urlCount": None,
                        "hashtagCount": None,
                        "retweetCount": None,
                        "join_date": tweet["user"]["join_date"],
                        "official": tweet["user"]["official"]
                        }

                    formatted_tweet = {
                        "tweet_id": int(tweet['id']), 
                        "text": self.utils.remove_emoji(tweet['body']), 
                        "user": tweet['user']['username'],
                        "create_date": calendar.timegm(datetime.strptime(tweet['created_at'],'%Y-%m-%dT%H:%M:%SZ').timetuple())*1000,
                        "retrieve_date": self.utils.getCurrentTimestsamp(),
                        "platform": 'stocktwits', 
                        "asset_id": asset["asset_id"], 
                        "metadata": metadata
                        }
                    if tweet['entities']['sentiment'] is not None:
                        formatted_tweet['sentiment'] = tweet['entities']['sentiment']['basic']
                    else:
                        formatted_tweet['sentiment'] = None

                    is_reliable = self.get_tweet_reliability(metadata)
                    if is_reliable == 1:
                        tweets_list.append(formatted_tweet)
                    
                    if len(tweets_list) > 0:
                        tweets_list.reverse()
                        new_tweet_list = self.db_manager.remove_duplicate_tweets(tweets_list, platform)
        except Exception as e:
            print("====================" + str(e) + "====================")
        return new_tweet_list

    def get_tweet_reliability(self, tweet_metadata_dict):
        predicted_reliability = 0 
        try:
            ''' load classifier '''
            os.path.join(os.getcwd(), "app", "utilities", "RF_classifier.pkl")
            RF_clf = joblib.load(os.path.join(os.getcwd(), "app", "utilities", "RF_classifier.pkl")) 
            
            ''' Create features to match training set '''    
            tweet_metadata_dict['user_verified'] = 1 if tweet_metadata_dict['user_verified']== True else 0
            
            ''' 'official' in stocktwits is similar to 'user_verified' in tweets '''
            if 'official' in tweet_metadata_dict:
                if (tweet_metadata_dict['official']):
                    tweet_metadata_dict['user_verified'] = 1
                else:
                    tweet_metadata_dict['user_verified'] = 0
        
            ''' to deal with stocktwits null values '''
            feature_names = ['followers', 'status_count', 'friends']
            for feature in feature_names:     
                if tweet_metadata_dict[feature] == None :
                    tweet_metadata_dict[feature] = 0

            ''' to check 1) if this works better (mainly regarding stockwits)'''
            # mean_status_count = 58995
            # if (tweet_metadata_dict['status_count']==0):
            #     tweet_metadata_dict['status_count'] = mean_status_count
            # 
            ''' to check 2) add the feature 'ages' ''' 
        
            ''' Reorder the dictionary key-value pairs '''
            key_order = ('user_verified', 'followers', 'status_count', 'friends')
            
            new_item = collections.OrderedDict()
            for k in key_order:
                new_item[k] = tweet_metadata_dict[k]
        
            ''' the tweet in the necessary format '''
            tweet_item = pd.Series(new_item) 
            predicted_reliability = RF_clf.predict([tweet_item])

        except Exception as e:
            print(str(e))
        finally:
            return predicted_reliability[0]
