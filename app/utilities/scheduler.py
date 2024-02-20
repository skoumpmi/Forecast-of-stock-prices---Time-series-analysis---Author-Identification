
from app.database.authManager import AuthManager
from types import prepare_class
from app.utilities.generalUtils import GeneralUtils
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers import interval, cron
import atexit

from ..streamer.assetIndicesStreamer import AssetIndicesStreamer
from ..portfolio.portfolioOverview import PortfolioOverview
from ..database.dbManager import DBManager
from ..streamer.tweetsStreamer import TwitterStreamer
from ..streamer.newsStreamer import NewsStreamer
from ..sentiment.SentimentAnalysis import SentimentAnalysis
from ..signals.signalsGenerator import SignalsGenerator
from ..reliability.reliabilityCalculator import ReliabilityCalculator
from ..entity_detection.context_dimension.contextDimensionClass import contextDimensionClass
from ..entity_detection.context_dimension.dictionaryGeneration import DictionaryGeneration
from ..entity_detection.entity_detection import EntityDetection

general_utils = GeneralUtils()

class Scheduler:

    def __init__(self):
        self.ai = AssetIndicesStreamer()
        self.po = PortfolioOverview()
        self.db = DBManager()
        self.ts = TwitterStreamer()
        self.sa = SentimentAnalysis()
        self.sg = SignalsGenerator()
        self.rc = ReliabilityCalculator()

    def runAssetsAndPorfoliosUpdateScheduler(self):
        assets_indices_downloaded, assetsAndPortfolioUpdated, assetsHistoryUpdated = False, False, False
        assets_indices_downloaded = self.ai.download_stock_indices_for_all_assets()
        if assets_indices_downloaded:
            assetsAndPortfolioUpdated = self.po.updateAssetsAndPortfolioPercentagesForAllUser()
            if assetsAndPortfolioUpdated:
                assetsHistoryUpdated = self.db.update_assets_and_portfolio_history_for_all_users()

        if assetsHistoryUpdated and  assetsAndPortfolioUpdated and assetsHistoryUpdated:
            print("-------- SUCCESS: Assets And Porfolios Update Scheduler completed--------")            
        else:
            print("-------- PROBLEM: Assets And Porfolios Update Scheduler NOT completed--------")

    def runSentimentAnalysisSignalsScheduler(self):
        new_tweets_updated, new_stocktwits_updated, new_articles_updated = False, False, False
        new_tweets_updated = self.ts.get_tweets_for_all_assets()
        if new_tweets_updated:
            new_stocktwits_updated = self.ts.get_stocktwits_for_all_assets()
            if new_stocktwits_updated:
                ns = NewsStreamer()
                new_articles_updated = ns.get_articles_for_all_assets()
        
        if new_articles_updated and new_tweets_updated and new_stocktwits_updated:
            self.sa.calculate_sentiment_for_all_assets()
            # self.sa.generate_sentiment_signal_for_all_assets()
            print("-------- SUCCESS: New Tweets, Stocktwits and Articles updated--------")
        else:
            print("-------- PROBLEM: New Tweets, Stocktwits and Articles NOT updated--------")

    def runReliabilityAnalysisScheduler(self):
        pass

    def runMLModelRetrainScheduler(self):
        pass
    
    def initialTrain(self):
        self.sg.train_ml_signals()
        self.sg.train_emergencies()
        self.sg.train_mixed_signals()

    def runDailyPipeline(self):
        self.runAssetsAndPorfoliosUpdateScheduler()
        # self.runSentimentAnalysisSignalsScheduler()
        # self.sg.generateSentimentSignals()
        # self.sg.generateMLSignalsAndEmergencies()
        # self.sg.generateMixedSignals()

    def runWeeklyPipeline(self):
        self.rc.update_source_and_writer_reliability()


    def downloadAssetIndecies(self):
        days = 32
        curr_timestamp = 1613654115000
        one_day = 86400000 
        for x in range(days):
            curr_timestamp = curr_timestamp + one_day
            general_utils.setCurrentTimestsamp(curr_timestamp)
            self.ai.download_stock_indices_for_all_assets()

    def simulatePortfolio(self):
        days = 5
        curr_timestamp = 1618229715000
        one_day = 86400000 
        user_id = 33
        for x in range(days):
            curr_timestamp = curr_timestamp + one_day
            general_utils.setCurrentTimestsamp(curr_timestamp)

            self.po.updateUserAssetsAndPortfolioPercentages(user_id)
            self.db.update_user_assets_history(user_id)
            self.db.update_portfolio_history(user_id)


    def run(self):
        pass
        # general_utils.setCurrentTimestsamp(1609766115000)
        # self.simulatePortfolio()
        # self.runDailyPipeline()
        # scheduler = BackgroundScheduler()
        # scheduler.add_job(id='scheduler_1', func=self.runDailyPipeline, trigger=cron.CronTrigger(hour=22, minute=0, second=0))
        # # scheduler.add_job(id='scheduler_2', func=self.runWeeklyPipeline, trigger=cron.CronTrigger(hour=22, minute=0, second=0))
        # scheduler.start()
        # atexit.register(lambda: scheduler.shutdown())
