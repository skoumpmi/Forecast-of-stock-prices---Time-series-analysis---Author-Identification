import os
import re
import pandas as pd
from ...entity_detection.context_dimension import cleanText
from ...entity_detection.context_dimension import docProcClass
from ...entity_detection.context_dimension import contextDimensionClass
from ...entity_detection.context_dimension import parseCSV
from ...database.dbManager import DBManager

class DictionaryGeneration:      
    def __init__(self): 
        self.basePath = os.path.join(os.getcwd(), 'app', 'entity_detection', 'context_dimension')
        self.contDimPath = os.path.join(self.basePath, 'context_dimensions_results')
        self.finalDictPath = os.path.join(os.getcwd(), 'app', 'entity_detection', 'dictionaries')

        self.contDim = contextDimensionClass.contextDimensionClass()
        self.allContextDimensionsIDList = self.contDim.allContextDimensionsIDList        

        self.db_manager = DBManager()

    def extractMostImportantWordsPerClass(self):
        for item in self.allContextDimensionsIDList:            
            asset_id = self.contDim.getConDimIDNo(item)            
            texts = self.db_manager.get_reports_by_asset_id(asset_id)
            categoryClass = contextDimensionClass.contextDimensionClass(item, len(self.allContextDimensionsIDList))
            
            if texts.size == 0:
                print("There is no articles downloaded for asset: " + item)                
            else:
                for index, row in texts.iterrows():
                    inputPDFFullFileName = ''    
                    docClass = docProcClass.docProcClass(inputPDFFullFileName)

                    text = row["full_text"]    
                    text = cleanText.textPreprocessing(text)
                    
                    retVal = docClass.loadTweet(text)
                    
                    if retVal:
                        retVal = docClass.scrappTextInSentences()
                        retVal = docClass.scrapSentencesInNamedEntities()
                        posPreservationList = [docClass.NAMED_ENTITY_LABEL_FW, docClass.NAMED_ENTITY_LABEL_IN, docClass.NAMED_ENTITY_LABEL_JJ, docClass.NAMED_ENTITY_LABEL_JJR, docClass.NAMED_ENTITY_LABEL_JJS, docClass.NAMED_ENTITY_LABEL_MD, docClass.NAMED_ENTITY_LABEL_NN, docClass.NAMED_ENTITY_LABEL_NNS, docClass.NAMED_ENTITY_LABEL_NNP, docClass.NAMED_ENTITY_LABEL_NNPS, docClass.NAMED_ENTITY_LABEL_RB, docClass.NAMED_ENTITY_LABEL_RBR, docClass.NAMED_ENTITY_LABEL_RBS, docClass.NAMED_ENTITY_LABEL_RP, docClass.NAMED_ENTITY_LABEL_SYM, docClass.NAMED_ENTITY_LABEL_VB, docClass.NAMED_ENTITY_LABEL_VBD, docClass.NAMED_ENTITY_LABEL_VBG, docClass.NAMED_ENTITY_LABEL_VBP, docClass.NAMED_ENTITY_LABEL_VBZ, docClass.NAMED_ENTITY_LABEL_WRB]
                    
                        retVal = docClass.filterNamedEntities(posPreservationList, True)
                        filteredNamedEntitiesList = docClass.getFilteredNamedEntitiesList()    
                            
                        #Main Text
                        listOfDisclaimerWords = ['Disclaimer', 'Disclosure']
                        mainTextFilteredNamedEntityList = cleanText.stopAtDisclamer(filteredNamedEntitiesList, listOfDisclaimerWords);    
                        filteredMainTextString = docClass.createStringFromNamedEntities(mainTextFilteredNamedEntityList);
                            
                        retVal = docClass.clearUniqueLabelsList()
                        #UniGrams - BiGrams - Trigrams
                        uniGramOccuranciesInText = docClass.getUnigramFreqTerms(filteredMainTextString, -1)
                        #biGramOccuranciesInText = docClass.getBigramsListFromText(filteredMainTextString, -1, 2)
                        #triGramOccuranciesInText = docClass.getTrigramsListFromText(filteredMainTextString, -1, 2)
                    
                        hypernymsList = docClass.createHypernymsList()
                        maxExistingHierarchyLevel = docClass.getMaxExistingHierarchyLayer()
                        entitiesList = docClass.retrieveCertainLayerFromHierarchy(-1)
                        condensedEntitiesList = docClass.condenseHypernymList(entitiesList)
                        NoW = docClass.getNoOfConceptualEntitiesInDoc()
                        
                        retVal = categoryClass.insertNewDocEntityFreqsList(condensedEntitiesList)
                        retVal = categoryClass.increaseNoW(NoW)
                        
                categoryEntityFreqs = categoryClass.condenseCategoryEntityList()
                categoryEntityProbs = categoryClass.getDocCategoryProbabilityList(True)
                fileName = item + '_' + str(categoryClass.increaseNoW(0)) + '.csv'
                outputFilesPath = os.path.join(self.contDimPath, fileName)                
                retVal = categoryClass.printContextDimensionProbFile(outputFilesPath)
        
    '''
        Scope: Function that extracts most significant words per asset (pe category) based on the
               related with each asset given documents
            
        Process: 1) read from database each asset's documents
                 2) extract most significant words and print them to files "extractMostImportantWordsPerClass()"
                 3) concatenate all significant words into one file (a.k.a dictionary) (def concatenateIntoOneFile())
                    - Certain the words can be filtered out
                    - The max numbers of words that will obtained for each asset can be specified
                 4) calculate prior probabilities for each word (def calculatePrioProbabilities())
        Output:  The output of the whole process is a unique file containing the relevant to each asset words 
                 coupled with their prior probabilities
                 This file can be used by the entities Detection module
    '''    
    def createEntitiesDetectionDictionary(self):
        # Create extract most important Words for each asset
        self.extractMostImportantWordsPerClass()        
        filter_out_keywords = ["https", "co", "said", "year"]
        max_num_of_entires=100
        self.concatenateIntoOneFile(max_num_of_entires, filter_out_keywords)        
        return
    
    def concatenateIntoOneFile(self, max_num_of_entires=None, filter_out_keywords_list=["https"], add_DB_asset_queries=True):
        # self.contDimPath

        tokens = pd.DataFrame(columns = ['internal_ticker', 'token', 'token_type'])
        for filename in os.listdir(self.contDimPath):
            if filename == "without_hashtags":
                continue
            print(filename)
            print(os.path.join(self.contDimPath, filename)) 
            filepath = os.path.join(self.contDimPath, filename)

            asset_words = pd.read_csv(filepath, names=["token","score"])
            internal_ticker = filename.split("_")[0]
            asset_words["internal_ticker"] = internal_ticker
            asset_words["token_type"] = "conDim"

            # remove certain Keywords
            asset_words = asset_words[~asset_words["token"].isin(filter_out_keywords_list)]

            # add search queries from database
            if add_DB_asset_queries:
                asset_info = self.db_manager.get_asset_by_ticker(internal_ticker)
                extra_keywords = []
                extra_keywords.append(asset_info["name"])
                extra_keywords.append(asset_info["ticker"])
                extra_keywords.append(asset_info["description"])
                extra_keywords.append(asset_info["yahoo_ticker"])
                twitter_query = asset_info["twitter_query"].split(",")
                twitter_query_strip = [i.strip() for i in twitter_query]
                extra_keywords.extend(twitter_query_strip)
                extra_keywords.append(asset_info["stocktwits_query"])
                print(extra_keywords)
                extra_keywords = [i for i in extra_keywords if i] #remove None values
                extra_keywords = [i.lower() for i in extra_keywords]
                extra_keywords_df = pd.DataFrame(extra_keywords, columns=["token"])
                extra_keywords_df["score"] = 1
                extra_keywords_df["internal_ticker"] = internal_ticker
                extra_keywords_df["token_type"] = "conDim"

                asset_words = pd.concat([extra_keywords_df, asset_words]).reset_index(drop=True)
                asset_words = asset_words.drop_duplicates(subset=['token'], ignore_index=True)

             
            # TODO: edw mporoume na epileksoume kai allous tropous px me vash to score kai na ta filtraroume
            # se periptwsh pou ta apotelesmata den einai ikanopoihtika
            if max_num_of_entires!=None:
                asset_words = asset_words.head(max_num_of_entires)
            
            asset_words = asset_words.drop(columns=["score"])

            tokens = tokens.append(asset_words, ignore_index=True)

        self.calculatePrioProbabilities(tokens)
        return
    
    def calculatePrioProbabilities(self, tokens_df):
        
        # ds = pd.read_csv('./Config/tokens_tweet_all_input.csv')
        n_ids = len(tokens_df['internal_ticker'].unique())
        n_tokens = len(tokens_df['token'].unique())

        unique_tokens = tokens_df.groupby(['token_type', 'token']).count().reset_index().rename(columns={'internal_ticker':'prior_probability'})#.drop(0, axis = 1)
        tokens_df = pd.merge(tokens_df, unique_tokens, on=['token_type', 'token'], how = 'left')
        tokens_df['prior_probability'] = 1 / tokens_df['prior_probability']
        tokens_df['word_count'] = tokens_df['token'].str.split().map(len)
        #ds.to_csv('./Config/tokens_tk.csv', sep = ',', float_format = '%.2f', index = False)
        # tokens_df['prob_new'] = tokens_df['prior_probability'] * (n_ids / n_tokens)

        finalDictName = os.path.join(self.finalDictPath, 'tokens_assets_all_output.csv')
        tokens_df.to_csv(finalDictName, index = False, sep = ',', float_format = '%.3f')
        return 
    
