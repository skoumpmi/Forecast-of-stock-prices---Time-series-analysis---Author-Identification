import pandas as pd
import numpy as np
import nltk
import re
import os.path
import tldextract
from collections import Counter
import datetime
from nltk.tokenize import TweetTokenizer
from ..database.dbManager import DBManager

class EntityDetection:    
    def __init__(self):
        self.posPreservationList = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBP', 'VBZ', 'WDT', 'WP',  'WP2$', 'WRB']
        self.db_manager = DBManager()
        self.tknzr = TweetTokenizer()

        # TODO: produce these files
        self.config_file_basename = 'tokens_assets_all_output.csv'
        self.basePath = os.path.join(os.getcwd(), 'app', 'entity_detection')
        self.dictionaryPath = os.path.join(self.basePath, 'dictionaries')
        self.dictionaryFilePathName = os.path.join(self.dictionaryPath, self.config_file_basename)
        self.tokens_dataset = pd.read_csv(self.dictionaryFilePathName)

        self.min_matching_probability = 0.2
        self.top_k_matches = 5

    def clean_text(self, text):
        raw_input = text
        raw_input = raw_input.replace('e.g.', 'eg')
        raw_input = raw_input.replace('n\'t', ' not')
        raw_input = raw_input.replace('Moody\'s', 'Moodys')
        raw_input = raw_input.replace('’', '')
        mainText = raw_input.lower().split('. ')
        mainText_raw = raw_input.lower()
        return mainText, mainText_raw

    def filterText(self, text):
        namedEntitiesList = []
        for sentence in text:
            parsedTokens = self.tknzr.tokenize(sentence)
            namedEntities = nltk.pos_tag(parsedTokens)
            namedEntitiesList.extend(namedEntities)
        filtered_entities = [x for x in namedEntitiesList if x[1] in self.posPreservationList]
        return filtered_entities

    def extractWeblinksFromRawText(self, rawText):
        weblinks = re.findall(r'[\w\.-]+@[\w\.-]+|www[\w.]+', rawText)
        tlds = [[tldextract.extract(x)[1], 1] for x in weblinks]
        matches = [x[1] for x in self.emails_to_int_tickers if (any(x[0] in y for y in tlds) and not pd.isnull(x[1]) )]
        matches += [x[1] for x in self.domains_to_int_tickers if (any(x[0] in y for y in tlds) and not pd.isnull(x[1]) )]
        return matches

    def processText(self, text, n_most_common_uni = -1, min_frequency_uni = 3, n_most_common_bi = -1, min_frequency_bi = 2, n_most_common_tri = -1, min_frequency_tri = 2):
        freqDistr = Counter(text.split(' '))
        freqDistr = dict(freqDistr.most_common(len(freqDistr) if n_most_common_uni == -1 else len(freqDistr) if n_most_common_uni > len(freqDistr) else n_most_common_uni))
        freqDistr = {x : freqDistr[x] for x in freqDistr if freqDistr[x] >= min_frequency_uni }
        if freqDistr:
            foundTickers = self.tokens_dataset[self.tokens_dataset['token'].isin(list(freqDistr.keys()))]
            if not foundTickers.empty:
                foundTickers['count'] = foundTickers.apply(lambda x: freqDistr[x['token']] if x['token'] in freqDistr else 0, axis = 1)

        bigrams = [' '.join(x for x in text.split(' ')[i:i+2]) for i in range(0, len(text)-1)]
        freqDistr_bigrams = Counter(bigrams)
        freqDistr_bigrams = dict(freqDistr_bigrams.most_common(len(freqDistr_bigrams) if n_most_common_bi == -1 else len(freqDistr_bigrams) if n_most_common_bi > len(freqDistr_bigrams) else n_most_common_bi))
        freqDistr_bigrams = {x : freqDistr_bigrams[x] for x in freqDistr_bigrams if freqDistr_bigrams[x] >= min_frequency_bi }
        if freqDistr_bigrams:
            foundTickers_bigrams = self.tokens_dataset[self.tokens_dataset['token'].isin(list(freqDistr_bigrams.keys()))]
            if not foundTickers_bigrams.empty:
                foundTickers_bigrams['count'] = foundTickers_bigrams.apply(lambda x: freqDistr_bigrams[x['token']] if x['token'] in freqDistr_bigrams else 0, axis = 1)

        trigrams = [' '.join(x for x in text.split(' ')[i:i+3]) for i in range(0, len(text)-2)]
        freqDistr_trigrams = Counter(trigrams)
        freqDistr_trigrams = dict(freqDistr_trigrams.most_common(len(freqDistr_trigrams) if n_most_common_tri == -1 else len(freqDistr_trigrams) if n_most_common_tri > len(freqDistr_trigrams) else n_most_common_tri))
        freqDistr_trigrams = {x : freqDistr_trigrams[x] for x in freqDistr_trigrams if freqDistr_trigrams[x] >= min_frequency_tri }
        if freqDistr_trigrams:
            foundTickers_trigrams = self.tokens_dataset[self.tokens_dataset['token'].isin(list(freqDistr_trigrams.keys()))]
            if not foundTickers_trigrams.empty:
                foundTickers_trigrams['count'] = foundTickers_trigrams.apply(lambda x: freqDistr_trigrams[x['token']] if x['token'] in freqDistr_trigrams else 0, axis = 1)

        foundTickers = pd.concat([foundTickers if 'foundTickers' in locals() else pd.DataFrame(), foundTickers_bigrams if 'foundTickers_bigrams' in locals() else pd.DataFrame(), foundTickers_trigrams if 'foundTickers_trigrams' in locals() else pd.DataFrame()])
        if not foundTickers.empty:
            foundTickers['max_token_lenght'] = foundTickers.apply(lambda x: self.tokens_dataset[self.tokens_dataset['token_type'] == x['token_type']]['word_count'].max(), axis = 1)
            foundTickers['matching_probability'] = foundTickers['prior_probability'] * foundTickers['count'] * (foundTickers['word_count'] / foundTickers['max_token_lenght'])
            #foundTickers['matching_probability'] = np.abs(np.log(foundTickers['prob_new'])) * foundTickers['count'] * (foundTickers['word_count'] / foundTickers['max_token_lenght'])
            tickerProbability = foundTickers.groupby(['internal_ticker'])['matching_probability'].sum().reset_index()
            tickerProbability.sort_values(by=['matching_probability'], inplace = True, ascending = False)
            return tickerProbability, foundTickers
        else:
            return None, None

    def detect_entity(self, text, source):
        disclaimerText = self.db_manager.get_source_disclaimer(source)
        mainText, mainText_raw = self.clean_text(text)
        weblink_tickers = []
        banned_tickers = None
        

        if disclaimerText is not None and mainText is not None:
            disclaimerText, disclaimerText_raw = self.clean_text(disclaimerText)
            fullText_raw = mainText_raw + ' ' + disclaimerText_raw
            mainText_filteredEntities = self.filterText(mainText)
            disclaimer_filteredEntities = self.filterText(disclaimerText)
            if source == "twitter" or source == "stocktwits":
                allowed_tickers, allowed_tickers_debug = self.processText(' '.join([x[0] for x in mainText_filteredEntities]), n_most_common_uni = 100, min_frequency_uni=1, min_frequency_bi=1, min_frequency_tri=1)
            else:
                allowed_tickers, allowed_tickers_debug = self.processText(' '.join([x[0] for x in mainText_filteredEntities]), n_most_common_uni = 100)
            banned_tickers, banned_tickers_debug = self.processText(' '.join([x[0] for x in disclaimer_filteredEntities]), n_most_common_tri = 5)
        elif mainText is not None:
            fullText_raw = mainText_raw
            mainText_filteredEntities = self.filterText(mainText)
            if source == "twitter" or source == "stocktwits":
                allowed_tickers, allowed_tickers_debug = self.processText(' '.join([x[0] for x in mainText_filteredEntities]), n_most_common_uni = 100, min_frequency_uni=1, min_frequency_bi=1, min_frequency_tri=1)
            else:
                allowed_tickers, allowed_tickers_debug = self.processText(' '.join([x[0] for x in mainText_filteredEntities]), n_most_common_uni = 100)
        else:
            print("None or null input text!")
            return None

        
        #TODO: na vgalw ta to_csvs kai na einai printable messages
        if allowed_tickers is not None:
            if banned_tickers is not None:
                banned_tickers['matching_probability_proc'] = banned_tickers.apply(lambda x:   0.5 * (x['matching_probability'] / banned_tickers['matching_probability'].max()) if x['internal_ticker'] in weblink_tickers else x['matching_probability'] / banned_tickers['matching_probability'].max(), axis = 1 )
                allowed_tickers = allowed_tickers[~allowed_tickers['internal_ticker'].isin(banned_tickers['internal_ticker'].values)]
            return allowed_tickers
        else:
            return None
           
        
    def check_entity(self, text, source, candidate_entity):
        result = self.detect_entity(text, source)
        if result is None or len(result)==0:
            return False
        else:
            result_final = result[result["matching_probability"] >= self.min_matching_probability] 
            result_final = result_final.head(self.top_k_matches)
            if candidate_entity in result_final["internal_ticker"].values:
                return True
            else:
                return False


