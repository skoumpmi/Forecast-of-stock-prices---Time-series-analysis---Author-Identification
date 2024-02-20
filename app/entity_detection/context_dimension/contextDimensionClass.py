import csv
import math
import pandas as pd
from nltk.corpus import wordnet as wn
from operator import itemgetter
from nltk.stem.snowball import SnowballStemmer
from ...database.dbManager import DBManager

class contextDimensionClass:
    
    NoW = 0
    categoryEntityList = []
    categoryProbabilityList = []
    ContextDimensionID = ''
    noOfAllConDims = 10
    
    def __init__(self, contextDimID = 'N/A', amountOfAllPossibleConDims = 10):
        
        self.NoW = 0
        self.categoryEntityList = []
        self.categoryProbabilityList = []
        self.ContextDimensionID = contextDimID
        self.noOfAllConDims = amountOfAllPossibleConDims
        
        db_manager = DBManager()
        self.all_assets = pd.DataFrame(db_manager.get_all_assets())        
        self.allContextDimensionsIDList = self.all_assets["ticker"].tolist()
        
    
    def getConDimIDNo(self, conDimID):
        res = self.all_assets[self.all_assets["ticker"]==conDimID]['asset_id'].values
        if len(res) > 0:
            return res[0]
        else:
            return -1
        
    def getConDimIDName(self, conDimIDNo):
        res = self.all_assets[self.all_assets["asset_id"]==conDimID]['ticker']
        if len(res) > 0:
            return res[0]
        else:
            return ''        

    def increaseNoW(self, newNoW):
        self.NoW = self.NoW + newNoW
        return self.NoW;
        
    def insertNewDocEntityFreqsList(self, newDocProbsFreq):

        initSize = len(self.categoryEntityList)
        
        for row in newDocProbsFreq:
            self.categoryEntityList.append(row)
            
        finalSize = len(self.categoryEntityList)
        
        diff = finalSize - initSize
        
        if diff >= 0:
            return self.categoryEntityList;
        else:
            return [];

    def condenseCategoryEntityList(self, printFlag = False):
        
        self.uniqueEntityListWithFreqs = []
        if len(self.categoryEntityList) == 0:
            return self.uniqueEntityListWithFreqs;
            
        print(len(self.categoryEntityList))
        print(len(self.categoryEntityList[0]))
        
        uniqueSynsetList = []
        for row in self.categoryEntityList:
            uniqueSynsetList.append(row[0])
            
        uniqueSynsetList = list(set(uniqueSynsetList))
        
        for uniqueSynset in uniqueSynsetList:
            aggregatedScore = 0
            for hypernymWithScore in self.categoryEntityList:
                if uniqueSynset == hypernymWithScore[0]:
                    aggregatedScore = aggregatedScore + hypernymWithScore[1]
                    extraLine = []
                    for itemIt in range(2, len(self.categoryEntityList[0])):
                        extraLine.append(hypernymWithScore[itemIt])
                    if printFlag:
                        print(uniqueSynset + ' ' + ' ' + str(aggregatedScore))
                        print(extraLine)
            self.uniqueEntityListWithFreqs.append([uniqueSynset, int(aggregatedScore)] + extraLine)                        
                
        self.uniqueEntityListWithFreqs = sorted(self.uniqueEntityListWithFreqs, key=itemgetter(1), reverse=True)
            
        return self.uniqueEntityListWithFreqs;
        
    def getDocCategoryProbabilityList(self, justEntitiesFlag = False):

        self.categoryProbabilityList = []
        if self.uniqueEntityListWithFreqs and self.NoW>0:
            for entityLine in self.uniqueEntityListWithFreqs:
                if justEntitiesFlag:
                    entity = entityLine[0]
                else:
                    entity = entityLine[0].lemmas()[0].name()
                entityFrequency = entityLine[1]
                
                extraLine = []
                for itemIt in range(2, len(entityLine)):
                    extraLine.append(entityLine[itemIt])
                self.categoryProbabilityList.append([entity, float(entityFrequency/self.NoW)] + extraLine)

        return self.categoryProbabilityList;

    def getEntityProbabilityList(self):
        return self.categoryProbabilityList

    def getEntityFrequencyList(self):
        return self.uniqueEntityListWithFreqs

    def printContextDimensionProbFile(self, csvOutputFileName):
    
        retVal = False
        
        if not self.categoryProbabilityList:
            return retVal;
            
        with open(csvOutputFileName, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            
            for row in self.categoryProbabilityList:
                try:
                    retVal = True
                    spamwriter.writerow([str(row[0]), round(row[1],5)])
                except ValueError:
                    retVal = False

        if retVal:
            print ('NOTE:Entity Probabilities for the Context Dimension ' + self.ContextDimensionID + ' CSV File has been printed in ' + csvOutputFileName)
            
        return retVal;

    def printContextDimensionEnhancedProbFile(self, csvOutputFileName):
    
        retVal = False
        
        if not self.categoryProbabilityList:
            return retVal;
            
        with open(csvOutputFileName, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            
            for row in self.categoryProbabilityList:
                try:
                    retVal = True
                    #condensedList.append([word1, aggregatedScore, pos, correctSynset, correctSynsetWord, stem1, hypernym1])
                    spamwriter.writerow([str(row[0]), round(row[1],5), str(row[2]), str(row[3]), str(row[4]), str(row[5]), str(row[6])])
                except ValueError:
                    retVal = False

        if retVal:
            print ('NOTE:Entity Probabilities for the Context Dimension ' +  self.ContextDimensionID + ' CSV File has been printed in ' + csvOutputFileName)
            
        return retVal;

    def printContextDimensionFreqFile(self, csvOutputFileName):
    
        retVal = False
        
        if not self.uniqueEntityListWithFreqs:
            return retVal;
            
        with open(csvOutputFileName, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            
            for row in self.uniqueEntityListWithFreqs:
                retVal = True
                spamwriter.writerow([str(row[0]), round(row[1],5)])

        if retVal:
            print ('NOTE:Entity Frequencies for the Context Dimension ' + self.ContextDimensionID + ' CSV File has been printed in ' + csvOutputFileName)
            
        return retVal;

    def parseContextDimensionProbFile(self, csvFileName, printOutputFlag=False, wordEntityColumnID = 0):
    
        retVal = False
        if printOutputFlag:
            print('NOTE:Parsing CSV File...')
        
        self.categoryProbabilityList = []
        with open(csvFileName) as csvfile:
            parsedCSVFile = csv.reader(csvfile, delimiter=',')
            for row in parsedCSVFile:
                if row[wordEntityColumnID] and row[1]:
                    self.categoryProbabilityList.append([row[wordEntityColumnID], row[1]])
                    retVal = True
                    if printOutputFlag:
                        print(str(row[wordEntityColumnID]) + ' ' + str(row[1]))
                else:
                    print('ERROR: ' + str(row[wordEntityColumnID]) + ' ' + str(row[1]))
                    continue
    
        if printOutputFlag:
            if retVal and self.categoryProbabilityList:
                print('NOTE:File ' + csvFileName + ' has been parsed and the corresponding internal Variable has been updated.')
            else:
                print('WARNING: Error in parsing file: ' + csvFileName)
            
        return retVal;

    def parseContextDimensionFreqFile(self, csvFileName, wordEntityColumn = 0, printOutputFlag=False):
    
        retVal = False
        if printOutputFlag:
            print('NOTE:Parsing CSV File...')
        
        self.uniqueEntityListWithFreqs = []
        with open(csvFileName) as csvfile:
            parsedCSVFile = csv.reader(csvfile, delimiter=',')
            for row in parsedCSVFile:
                if row[wordEntityColumn] and row[1]:
                    self.uniqueEntityListWithFreqs.append([row[wordEntityColumn], row[1]])
                    if printOutputFlag:
                        print(str(row[wordEntityColumn]) + ' ' + str(row[1]))
                else:
                    retVal = False
                    break
    
        if printOutputFlag:
            if retVal:
                print('NOTE:File ' + csvFileName + ' has been parsed and the corresponding internal Variable has been updated.')
            else:
                print('WARNING: Error in parsing file: ' + csvFileName)
            
        return retVal;

    def getWordnetSimilarity(self, distribution1, distribution2, synsetEntityFlag1 = False, synsetEntityFlag2 = False, debugFlag = False):
        
        debugInformationList = []

        symmetricDistanceScore = 0
        
        for wordIt in range(0, len(distribution1)):
            
            wordline1 = distribution1[wordIt]
            docEntityProb1 = float(wordline1[1])            
            if synsetEntityFlag1:
                docSynset1 = wn.synset(wordline1[0])
                docEntity1 =wordline1[0]
            else:
                docSynset1 = wn.synsets(wordline1[0])[0]
                docEntity1 = docSynset1.lemmas()[0].name()
                        
            indexNo = [(k, detectedEntity.index(docEntity1))  for k, detectedEntity in enumerate(distribution2) if docEntity1 == detectedEntity[0]]
            if not indexNo:
                if debugFlag:
                    debugInformationList.append(['WARNING: Word ' + str(docEntity1) +  'was not found in  Distribution #2...'])
                continue

            wordline2 = distribution2[indexNo[0][0]]
            docEntityProb2 = float(wordline2[1])
            if synsetEntityFlag2:
                docSynset1 = wn.synset(wordline1[0])
            else:
                docSynset2 = wn.synsets(wordline2[0])[0]


            addedRE = docSynset1.path_similarity(docSynset2)#*docEntityProb1*docEntityProb2
            symmetricDistanceScore = symmetricDistanceScore + (addedRE)
                
            if debugFlag:
                debugInformationList.append([docEntity1, docEntityProb1, docEntityProb2, addedRE, symmetricDistanceScore])

        finalDistance = symmetricDistanceScore 
        
        return [finalDistance, debugInformationList]        

    def getAdvancedKLDistance(self, distribution1, distribution2, NoW1 = -1, considerIGValsFlag=False, debugFlag = False):

        stemmer = SnowballStemmer("english")        
        
        debugInformationList = []

        relativeEntropy12 = 0
        relativeEntropy21 = 0
        symmetricRelativeEntropy = 0
        detominator = 0.00000000001
        #sumD1 = 0
        #sumD2 = 0
        
        
        indexDistList2 = [0 for x in range(len(distribution2))] 
        for wordIt in range(0, len(distribution1)):
            wordLine1 = distribution1[wordIt]
            word1 = wordLine1[0]
            wordStem1 = stemmer.stem(word1)
            wordProb1 = float(wordLine1[1])
            wordIGVal1 =  float(wordLine1[2])

            indexNo1 = [(k, detectedEntity.index(word1))  for k, detectedEntity in enumerate(distribution2) if word1 == detectedEntity[0]]
            indexNo2 = [(k, detectedEntity.index(wordStem1)) for k, detectedEntity in enumerate(distribution2) if wordStem1 == detectedEntity[0]]
            if not indexNo1 and not indexNo2:
                if debugFlag:
                    debugInformationList.append(['WARNING: Word ' + str(word1) +  'was not found in  Distribution #2...'])
                #print('WARNING: Word ' + str(word1) +  'was not found in  Distribution #2...')
                continue                        
                 
            if indexNo1:
                if indexDistList2[indexNo1[0][0]] == 1:
                    #print(word1 + ' has been skipped...')
                    continue
                wordLine2 = distribution2[indexNo1[0][0]]
                indexDistList2[indexNo1[0][0]] == 1
            else:
                if indexDistList2[indexNo2[0][0]] == 1:
                    #print(word1 + ' has been skipped...')
                    continue
                wordLine2 = distribution2[indexNo2[0][0]]
                indexDistList2[indexNo2[0][0]] == 1
            wordProb2 = float(wordLine2[1])
            
            if considerIGValsFlag:
                wordProb2 = wordIGVal1*wordProb2
                wordProb2 = wordIGVal1*wordProb2
                
            addedRE1 = 0
            addedRE2 = 0

            if wordProb1 == 0:
                addedRE1 = 0;
            else:
                if wordProb2 == 0:
                    addedRE1 = wordProb1*math.log(wordProb1/detominator)
                else:
                    addedRE1 = wordProb1*math.log(wordProb1/wordProb2)
            

            if wordProb2 == 0:
                addedRE2 = 0;
            else:
                if wordProb1 == 0:
                    addedRE2 = wordProb2*math.log(wordProb2/detominator)
                else:
                    addedRE2 = wordProb2*math.log(wordProb2/wordProb1)

            addedRE = (addedRE1 + addedRE2)/2
            relativeEntropy12 = relativeEntropy12 + addedRE1
            relativeEntropy21 = relativeEntropy21 + addedRE2
            symmetricRelativeEntropy = symmetricRelativeEntropy + addedRE
            #print(word1, word2, wordProb1, wordProb2, wordIGVal1, symmetricRelativeEntropy)
                
            if debugFlag:
                debugInformationList.append([word1, wordProb1, wordProb2, addedRE, symmetricRelativeEntropy])

        finalDistance = -1
        if NoW1 == -1:
            finalDistance = symmetricRelativeEntropy 
        else:
            finalDistance = symmetricRelativeEntropy - math.log(1/self.noOfAllConDims)/NoW1
        
        return [finalDistance, debugInformationList]

    def getKLDistanceForUniqueConDims(self, distribution1, distribution2, wordEntityColumn = 0, NoW1 = -1, printFlag = False, debugFlag = False):
        
        debugInformationList = []

        relativeEntropy12 = 0
        relativeEntropy21 = 0
        symmetricRelativeEntropy = 0
        detominator = 0.00000000001
        symmetricRelativeEntropy = 0
        indexDistList2 = [0 for x in range(len(distribution2))] 
        for wordIt in range(0, len(distribution1)):
            wordLine1 = distribution1[wordIt]
            word1 = wordLine1[wordEntityColumn]
            #wordStem1 = stemmer.stem(word1)
            wordProb1 = float(wordLine1[1])

            indexNo1 = [(k, detectedEntity.index(word1))  for k, detectedEntity in enumerate(distribution2) if word1 == detectedEntity[0]]
            if not indexNo1:
                if debugFlag:
                    debugInformationList.append(['WARNING: Word ' + str(word1) +  'was not found in  Distribution #2...'])
                #print('WARNING: Word ' + str(word1) +  'was not found in  Distribution #2...')
                continue                        
            
            if indexDistList2[indexNo1[0][0]] == 1:
                #print(word1 + ' has been skipped...')
                continue
            wordLine2 = distribution2[indexNo1[0][0]]
            indexDistList2[indexNo1[0][0]] == 1
            wordProb2 = float(wordLine2[1])
            if printFlag:
                print([word1, wordProb1, wordProb2])
                
            symmetricRelativeEntropy = symmetricRelativeEntropy + abs(1*wordProb2*math.log(wordProb2/wordProb1))
            '''    
            addedRE1 = 0
            addedRE2 = 0

            if wordProb1 == 0:
                addedRE1 = 0;
            else:
                if wordProb2 == 0:
                    addedRE1 = wordProb1*math.log(wordProb1/detominator)
                else:
                    addedRE1 = wordProb1*math.log(wordProb1/wordProb2)
            

            if wordProb2 == 0:
                addedRE2 = 0;
            else:
                if wordProb1 == 0:
                    addedRE2 = wordProb2*math.log(wordProb2/detominator)
                else:
                    addedRE2 = wordProb2*math.log(wordProb2/wordProb1)

            addedRE = (addedRE1 + addedRE2)/2
            relativeEntropy12 = relativeEntropy12 + addedRE1
            relativeEntropy21 = relativeEntropy21 + addedRE2
            symmetricRelativeEntropy = symmetricRelativeEntropy + addedRE
            #print(word1, word2, wordProb1, wordProb2, wordIGVal1, symmetricRelativeEntropy)
                
            if debugFlag:
                debugInformationList.append([word1, wordProb1, wordProb2, addedRE, symmetricRelativeEntropy])

        finalDistance = -1
        if NoW1 == -1:
            finalDistance = symmetricRelativeEntropy 
        else:
            finalDistance = symmetricRelativeEntropy - math.log(1/self.noOfAllConDims)/NoW1
        '''
        finalDistance = symmetricRelativeEntropy
        return [finalDistance, debugInformationList]
    
    def getKLDistance(self, distribution1, distribution2, IGList = [], NoW1 = -1, synsetEntityFlag1 = False, debugFlag = False):
        
        debugInformationList = []

        relativeEntropy12 = 0
        relativeEntropy21 = 0
        symmetricRelativeEntropy = 0
        detominator = 0.00000000001
        #sumD1 = 0
        #sumD2 = 0
        
        for wordIt in range(0, len(distribution1)):
            word1 = distribution1[wordIt]

            docEntityProb1 = float(word1[1])                        
            if synsetEntityFlag1:
                docEntity1 = word1[0].lemmas()[0].name()
            else:
                docEntity1 = word1[0]
                
            if IGList:
                iGIndexNo = [(k, detectedEntity.index(docEntity1))  for k, detectedEntity in enumerate(IGList) if docEntity1 == detectedEntity[0]]
                if not iGIndexNo:
                    continue;
                
            indexNo = [(k, detectedEntity.index(docEntity1))  for k, detectedEntity in enumerate(distribution2) if docEntity1 == detectedEntity[0]]
            if not indexNo:
                if debugFlag:
                    debugInformationList.append(['WARNING: Word ' + str(docEntity1) +  'was not found in  Distribution #2...'])
                continue
                                    
            word2 = distribution2[indexNo[0][0]]
            docEntityProb2 = float(word2[1])                
            
            #sumD1 = sumD1 + float(word1[1])
            #sumD2 = sumD2 + float(word2[1])
            
            addedRE1 = 0
            addedRE2 = 0

            if docEntityProb1 == 0:
                addedRE1 = 0;
            else:
                if docEntityProb2 == 0:
                    addedRE1 = docEntityProb1*math.log(docEntityProb1/detominator)
                else:
                    addedRE1 = docEntityProb1*math.log(docEntityProb1/docEntityProb2)
            

            if docEntityProb2 == 0:
                addedRE2 = 0;
            else:
                if docEntityProb1 == 0:
                    addedRE2 = docEntityProb2*math.log(docEntityProb2/detominator)
                else:
                    addedRE2 = docEntityProb2*math.log(docEntityProb2/docEntityProb1)

            addedRE = (addedRE1 + addedRE2)/2
            relativeEntropy12 = relativeEntropy12 + addedRE1
            relativeEntropy21 = relativeEntropy21 + addedRE2
            symmetricRelativeEntropy = symmetricRelativeEntropy + addedRE
                
            if debugFlag:
                debugInformationList.append([docEntity1, docEntityProb1, docEntityProb2, addedRE, symmetricRelativeEntropy])

        finalDistance = -1
        if NoW1 == -1:
            finalDistance = symmetricRelativeEntropy 
        else:
            finalDistance = symmetricRelativeEntropy - math.log(1/self.noOfAllConDims)/NoW1
        
        return [finalDistance, debugInformationList]
        
    def getKLDistance_new(self, distribution1, distribution2, wordEntityColumnID = 0, IGList = [], NoW1 = -1, printFlag = False, debugFlag = False):
        
        debugInformationList = []

        relativeEntropy12 = 0
        relativeEntropy21 = 0
        symmetricRelativeEntropy = 0
        detominator = 0.00000000001
        #sumD1 = 0
        #sumD2 = 0
        
        for wordIt in range(0, len(distribution1)):
            word1 = distribution1[wordIt]

            docEntityProb1 = float(word1[1])                        
            docEntity1 = word1[wordEntityColumnID].lower()
                
            if IGList:
                iGIndexNo = [(k, detectedEntity.index(docEntity1))  for k, detectedEntity in enumerate(IGList) if docEntity1 == detectedEntity[wordEntityColumnID]]
                if not iGIndexNo:
                    continue;
            
            try:
                indexNo = [(k, detectedEntity.index(docEntity1))  for k, detectedEntity in enumerate(distribution2) if docEntity1 == detectedEntity[0]]
            except ValueError:
                indexNo = []
                
            if not indexNo:
                if debugFlag:
                    debugInformationList.append(['WARNING: Word ' + str(docEntity1) +  'was not found in  Distribution #2...'])
                continue
                                    
            word2 = distribution2[indexNo[0][0]]
            docEntityProb2 = float(word2[1])                
            
            addedRE1 = 0
            addedRE2 = 0
            addedRE = 0

            if docEntityProb1 == 0:
                addedRE1 = 0;
            else:
                if docEntityProb2 == 0:
                    addedRE1 = docEntityProb1*math.log(docEntityProb1/detominator)
                else:
                    addedRE1 = docEntityProb1*math.log(docEntityProb1/docEntityProb2)
            

            if docEntityProb2 == 0:
                addedRE2 = 0;
            else:
                if docEntityProb1 == 0:
                    addedRE2 = docEntityProb2*math.log(docEntityProb2/detominator)
                else:
                    addedRE2 = docEntityProb2*math.log(docEntityProb2/docEntityProb1)

            addedRE = (addedRE1 + addedRE2)/2
            relativeEntropy12 = relativeEntropy12 + addedRE1
            relativeEntropy21 = relativeEntropy21 + addedRE2
            symmetricRelativeEntropy = symmetricRelativeEntropy + docEntityProb2*docEntityProb1
            
            if printFlag:
                print([docEntity1, docEntityProb1, docEntityProb2, addedRE])

            #symmetricRelativeEntropy = symmetricRelativeEntropy + docEntityProb1
            
            if debugFlag:
                debugInformationList.append([docEntity1, docEntityProb1, docEntityProb2, addedRE, symmetricRelativeEntropy])

        finalDistance = -1
        if NoW1 == -1:
            finalDistance = symmetricRelativeEntropy 
        else:
            finalDistance = symmetricRelativeEntropy - math.log(1/self.noOfAllConDims)/NoW1
        
        return [finalDistance, debugInformationList]        

    def getMatchingProbabilityWithDoc(self, newDocProbsFreq, newDocNoW, synsetEntityFlag = False):
        
        retVal = -1
        if not self.categoryProbabilityList:
            print('WARNING: No Context Dimension Probabilities have been loaded...')
            return retVal;
        
        if self.noOfAllConDims <= 0:
            print('WARNING: Check Amount (variable) of Context Dimensions...')
            return retVal;

        relativeEntropyScore = 0
        for newDocRow in newDocProbsFreq:
            if synsetEntityFlag:
                docEntity = newDocRow[0].lemmas()[0].name()
            else:
                docEntity = newDocRow[0]
                
            indexNo = [(k, detectedEntity.index(docEntity))  for k, detectedEntity in enumerate(self.categoryProbabilityList) if docEntity == detectedEntity[0]]
            if indexNo:
                docEntityProb = newDocRow[1]/newDocNoW
                conDimEntityProb = float(self.categoryProbabilityList[indexNo[0][0]][1])
                if docEntityProb == 0 or conDimEntityProb == 0:
                    relativeEntropyScore = relativeEntropyScore + 0
                else:
                    relativeEntropyScore = relativeEntropyScore + docEntityProb*math.log(docEntityProb/conDimEntityProb)
                
            
        if relativeEntropyScore > 0:
            relativeEntropyScore = relativeEntropyScore - math.log(1/self.noOfAllConDims)/newDocNoW
            return relativeEntropyScore;
        else:
            retVal = 0;
            print('WARNING: No Matching Entities between the provided Document and Context Dimension: ' + self.ContextDimensionID)
            return retVal;
