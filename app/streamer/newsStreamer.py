from app.database.dbManager import DBManager
from app.utilities.generalUtils import GeneralUtils
from app.entity_detection.entity_detection import EntityDetection
import configparser
from newsapi import NewsApiClient
from datetime import datetime
from datetime import timedelta
from newspaper import fulltext, Article
import requests
import spacy
from bs4 import BeautifulSoup as bs

config = configparser.ConfigParser()
config.read('config.ini')
not_queryable_tickers=['BABA', 'C', 'FB', 'F', 'KO', 'V', 'GE', 'PG']

class NewsStreamer:

    def __init__(self):
        
        self.domains = config['newsapi']['domains']
        self.newsapi = NewsApiClient(api_key=config['newsapi']['token'])
        self.db_manager = DBManager()
        self.nlp = spacy.load("en_core_web_md")
        self.similarity_threshold = float(config['news_streamer']['similarity_threshold'])
        self.utils = GeneralUtils()
        self.ed = EntityDetection()

    def get_articles_for_all_assets(self):
        success = False
        try:
            assets = self.db_manager.get_all_assets()
            
            from_timestamp, to_timestamp = self.utils.getTodayStartEndTimestamps()
            print("-------- Get New Articles --------")
            for asset in assets:
                print("Asset Name: " + str(asset["name"]))

                filtered_articles = self.get_articles_for_asset(asset, from_timestamp/1000, to_timestamp/1000)
                print("Articles Number: " + str(len(filtered_articles)))
                
                if len(filtered_articles) > 0:
                    success_save = self.db_manager.insert_news(filtered_articles)
                    print("Stored: " + str(success_save))
                print("-------------------------------")
            success = True
        except Exception as e:
            success = False
            print("-------- PROBLEM: New Articles NOT updated--------")
            print("====================" + str(e) + "====================")
        finally:
            return success

    def get_articles_for_asset(self, asset, from_timestamp, to_timestamp):
        result = []
        try:
            if asset["ticker"] in not_queryable_tickers:
                query = '"'+asset["name"]+'"'
            else:
                query = '"' + asset["ticker"] + '" OR "' + asset["name"] + '"'
            asset_id = asset["asset_id"]
            response = self.newsapi.get_everything(
                q=query, 
                domains=self.domains, 
                from_param= from_timestamp,
                to=to_timestamp,
                language='en', 
                sort_by='publishedAt',  
                page=1, 
                page_size= 100)
            
            if len(response['articles']) > 0:
                articles_list = []
                for article in response['articles']:
                    try:
                        if article['source']['id'] == 'the-wall-street-journal':
                            html = requests.get(article["url"],
                                                headers={'User-Agent': 'Popular browser\'s user-agent'}).text
                            text = fulltext(html)
                        elif article['source']['id'] == 'bbc-news':
                            try:
                                parsed = BBC(article["url"])
                                text = parsed.body
                            except:
                                continue
                        else:
                            art = Article(article["url"])
                            art.download()
                            art.parse()
                            text = art.text
                        error_message = "Please make sure your browser supports JavaScript and cookies and that you are not blocking them from loading. For more information you can review our Terms of Service and Cookie Policy."
                        if error_message in text:
                            continue
                        article["content"] = text.encode('ascii', 'ignore').decode('ascii')
                        articles_list.append((article, asset_id))
                    except Exception as e:
                        print(str(e))
                articles_list_filtered = self.filter_articles_simple(articles_list)
                result = self.check_entity_for_all_articles(articles_list_filtered, asset)
        except Exception as e:
            print(str(e))
        finally:
            return result
     
    def filter_articles_with_spacy(self, all_assets_articles):
        print('Number of parsed articles: ', len(all_assets_articles))
        articles_to_save = []
        black_list = [] # a black list of article indices, that won't be saved to the db
        for idx_parsed, article_parsed in enumerate(all_assets_articles): # Loop all parsed articles
            print('Parsed article index: ', idx_parsed)
            if idx_parsed in black_list: # If the article's index is in the black list, just continue looping (thus don't save)
                continue
            # Loop all parsed articles again, to find similar between parsed articles
            for idx_parsed2, article_parsed2 in enumerate(all_assets_articles):
                # If you find any two similar
                if self.nlp(article_parsed2[0]["content"]).similarity(self.nlp(article_parsed[0]["content"])) > self.similarity_threshold:
                    if idx_parsed != idx_parsed2: # Check that it didn't find similar its self
                        black_list.append(idx_parsed2) # Add its doppelganger into the black list (so it will be exluded)
            # If current article was not in the black list neither in the database, then save it !
            articles_to_save.append(article_parsed)
        return articles_to_save

    def filter_articles_simple(self, all_assets_articles):
        articles_to_save = []
        try:
            # print('Number of parsed articles: ', len(all_assets_articles))
            black_list = [] # a black list of article indices, that won't be saved to the db
            for idx_parsed, article_parsed in enumerate(all_assets_articles): # Loop all parsed articles
                # print('Parsed article index: ', idx_parsed)
                if idx_parsed in black_list: # If the article's index is in the black list, just continue looping (thus don't save)
                    continue
                # Loop all parsed articles again, to find similar between parsed articles
                for idx_parsed2, article_parsed2 in enumerate(all_assets_articles):
                    # If you find any two similar
                    if article_parsed2[0]["content"] == article_parsed[0]["content"]:
                        if idx_parsed != idx_parsed2: # Check that it didn't find similar its self
                            black_list.append(idx_parsed2) # Add its doppelganger into the black list (so it will be exluded)
                # If current article was not in the black list neither in the database, then save it !
                articles_to_save.append(article_parsed)
        except Exception as e:
            print(str(e))
        finally:
            return articles_to_save

    def check_entity_for_all_articles(self, articles_list_filtered, asset):
        result = []
        try:
            for article in articles_list_filtered:
                #Skip Entity Detection 
                # isEntityCorrect = True
                isEntityCorrect = self.ed.check_entity(article[0]["content"], article[0]["source"]["name"], asset["ticker"])
                if isEntityCorrect:
                    result.append(article)

        except Exception as e:
            print(str(e))
        finally:
            return result 

class BBC:
    def __init__(self, url: str):
        article = requests.get(url)
        self.soup = bs(article.content, "html.parser")
        self.body = self.get_body()

    def get_body(self) -> list:
        body = self.soup.find(id="root")
        article = body.findAll('article')[0]
        parag = [p.text for p in article]
        clean_parag = parag[2:-1]
        return ' '.join(clean_parag)
