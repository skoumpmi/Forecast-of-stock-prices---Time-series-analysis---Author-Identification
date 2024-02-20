# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:06:14 2015

@author: Anastasis
"""

# import parsePDF

import csv

import nltk
import nltk.data
import nltk.corpus

from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from operator import itemgetter



class docProcClass:

    '''Class extracting the document/text characteristics '''
 
    NAMED_ENTITY_LABEL_CC   = 'CC' 	    #Coordinating conjunction
    NAMED_ENTITY_LABEL_CD   = 'CD' 	    #Cardinal number
    NAMED_ENTITY_LABEL_DT   = 'DT' 	    #Determiner
    NAMED_ENTITY_LABEL_EX   = 'EX' 	    #Existential there
    NAMED_ENTITY_LABEL_FW   = 'FW' 	    #Foreign word
    NAMED_ENTITY_LABEL_IN   = 'IN' 	    #Preposition or subordinating conjunction
    NAMED_ENTITY_LABEL_JJ   = 'JJ' 	    #Adjective
    NAMED_ENTITY_LABEL_JJR  = 'JJR'	    #Adjective, comparative
    NAMED_ENTITY_LABEL_JJS  = 'JJS'	    #Adjective, superlative
    NAMED_ENTITY_LABEL_LS   = 'LS' 	    #List item marker
    NAMED_ENTITY_LABEL_MD   = 'MD' 	    #Modal
    NAMED_ENTITY_LABEL_NN   = 'NN' 	    #Noun, singular or mass
    NAMED_ENTITY_LABEL_NNS  = 'NNS'	    #Noun, plural
    NAMED_ENTITY_LABEL_NNP  = 'NNP'	    #Proper noun, singular
    NAMED_ENTITY_LABEL_NNPS = 'NNPS'	#Proper noun, plural
    NAMED_ENTITY_LABEL_PDT  = 'PDT' 	#Predeterminer
    NAMED_ENTITY_LABEL_POS  = 'POS' 	#Possessive ending
    NAMED_ENTITY_LABEL_PRP  = 'PRP' 	#Personal pronoun
    NAMED_ENTITY_LABEL_PRP2 = 'PRP$'	#Possessive pronoun
    NAMED_ENTITY_LABEL_RB   = 'RB' 	    #Adverb
    NAMED_ENTITY_LABEL_RBR  = 'RBR' 	#Adverb, comparative
    NAMED_ENTITY_LABEL_RBS  = 'RBS' 	#Adverb, superlative
    NAMED_ENTITY_LABEL_RP   = 'RP' 	    #Particle
    NAMED_ENTITY_LABEL_SYM  = 'SYM' 	#Symbol
    NAMED_ENTITY_LABEL_TO   = 'TO' 	    #to
    NAMED_ENTITY_LABEL_UH   = 'UH' 	    #Interjection
    NAMED_ENTITY_LABEL_VB   = 'VB' 	    #Verb, base form
    NAMED_ENTITY_LABEL_VBD  = 'VBD' 	#Verb, past tense
    NAMED_ENTITY_LABEL_VBG  = 'VBG' 	#Verb, gerund or present participle VBN Verb, past participle
    NAMED_ENTITY_LABEL_VBP  = 'VBP' 	#Verb, non-3rd person singular present
    NAMED_ENTITY_LABEL_VBZ  = 'VBZ' 	#Verb, 3rd person singular present
    NAMED_ENTITY_LABEL_WDT  = 'WDT' 	#Wh-determiner
    NAMED_ENTITY_LABEL_WP   = 'WP' 	    #Wh-pronoun
    NAMED_ENTITY_LABEL_WP2  = 'WP$' 	#Possessive wh-pronoun
    NAMED_ENTITY_LABEL_WRB  = 'WRB' 	#Wh-adverb
    
        
    inputFilePath = ''
    parsedPDFUnits = []
    scrappedSentencesList = []
    scrappedWordsList = []
    namedEntitiesList = []
    filteredNamedEntitiesList = []
    
    uniGramListWithProbs = [] 
    biGramListWithProbs = [] 
    triGramListWithProbs = [] 

    nonOverlappingTokensFoundFlag = False;
    cleanUnigrams = []
    cleanBigrams = []
    cleanTrigrams = []

    NoW = -1
    
    hypernymHierarchyList = []
    uniqueSynsetListWithScores = []
    entityProbabilityInText = []

    bagOfWords = []

    def __init__(self, inputFullFileName):        

        self.inputFilePath = inputFullFileName;

        self.parsedPDFUnits = []
        self.scrappedSentencesList = []
        self.scrappedWordsList = []
        self.namedEntitiesList = []
        self.filteredNamedEntitiesList = []

        self.uniGramListWithProbs = [] 
        self.biGramListWithProbs = [] 
        self.triGramListWithProbs = []         
        
        self.nonOverlappingTokensFoundFlag = False;
        self.cleanUnigrams = []
        self.cleanBigrams = []
        self.cleanTrigrams = []

        self.NoW = -1
        
        self.hypernymHierarchyList = []
        self.uniqueSynsetListWithScores = []
        self.entityProbabilityInText = []

        self.bagOfWords = []

        return;

#--------------NLTK - Wordnet POS Compatibility--------------------
    def convertNltkNN2WordnetSynset(self, nNType):
    
        retVal = ''
    
        if nNType == self.NAMED_ENTITY_LABEL_CC:	  #Coordinating conjunction
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_CD:    #Cardinal number
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_DT:    #Determiner
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_EX:    #Existential there
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_FW:    #Foreign word
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_IN:    #Preposition or subordinating conjunction
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_JJ:    #Adjective
            retVal = wn.ADJ
        elif nNType == self.NAMED_ENTITY_LABEL_JJR:   #Adjective, comparative
            retVal = wn.ADJ
        elif nNType == self.NAMED_ENTITY_LABEL_JJS:   #Adjective, superlative
            retVal = wn.ADJ
        elif nNType == self.NAMED_ENTITY_LABEL_LS:    #List item marker
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_MD:    #Modal
            retVal = wn.VERB
        elif nNType == self.NAMED_ENTITY_LABEL_NN:    #Noun, singular or mass
            retVal = wn.NOUN
        elif nNType == self.NAMED_ENTITY_LABEL_NNS:   #Noun, plural
            retVal = wn.NOUN
        elif nNType == self.NAMED_ENTITY_LABEL_NNP:   #Proper noun, singular
            retVal = wn.NOUN
        elif nNType == self.NAMED_ENTITY_LABEL_NNPS:  #Proper noun, plural
            retVal = wn.NOUN
        elif nNType == self.NAMED_ENTITY_LABEL_PDT:   #Predeterminer
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_POS:   #Possessive ending
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_PRP:   #Personal pronoun
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_PRP2:  #Possessive pronoun
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_RB:    #Adverb
            retVal = wn.ADV
        elif nNType == self.NAMED_ENTITY_LABEL_RBR:   #Adverb, comparative
            retVal = wn.ADV
        elif nNType == self.NAMED_ENTITY_LABEL_RBS:   #Adverb, superlative
            retVal = wn.ADV
        elif nNType == self.NAMED_ENTITY_LABEL_RP:    #Particle
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_SYM:   #Symbol
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_TO:    #to
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_UH:    #Interjection
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_VB:    #Verb, base form
            retVal = wn.VERB
        elif nNType == self.NAMED_ENTITY_LABEL_VBD:   #Verb, past tense
            retVal = wn.VERB
        elif nNType == self.NAMED_ENTITY_LABEL_VBG:   #Verb, gerund or present participle VBN Verb, past participle
            retVal = wn.VERB
        elif nNType == self.NAMED_ENTITY_LABEL_VBP:   #Verb, non-3rd person singular present
            retVal = wn.VERB
        elif nNType == self.NAMED_ENTITY_LABEL_VBZ:   #Verb, 3rd person singular present
            retVal = wn.VERB
        elif nNType == self.NAMED_ENTITY_LABEL_WDT:   #Wh-determiner
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_WP:    #Wh-pronoun
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_WP2:   #Possessive wh-pronoun
            retVal = 'NA'
        elif nNType == self.NAMED_ENTITY_LABEL_WRB:   #Wh-adverb
            retVal = 'NA'
        return retVal;
#----------------------------------

#--------------Parsing PDF(s)--------------------
    def loadTweet(self, text, printFlag = True):
        retVal = True
        self.parsedPDFUnits = []        
        
        self.parsedPDFUnits = [text] 
        if not self.parsedPDFUnits:
            retVal = False
            
        return retVal;        
    
    # def parsePDF(self, maxNoOfPages = -1, printFlag = False):
    #     retVal = False
    #     self.parsedPDFUnits = []        

    #     pdfParsingClass = parsePDF.parsePDFClass(self.inputFilePath)        
        
    #     retVal = pdfParsingClass.parsePDF(maxNoOfPages, printFlag)
    #     self.parsedPDFUnits = pdfParsingClass.getTextUnitsArray()
        
    #     return retVal;

    def getParsedPDFUnits(self):
        return self.parsedPDFUnits;

    def scrappTextInSentences(self, resourcesDataPackage = 'tokenizers/punkt/english.pickle'):
        retVal = False
        sent_detector = nltk.data.load(resourcesDataPackage)

        self.scrappedSentencesList = []
        parsedSentences = []
        
        for text in self.parsedPDFUnits:
            parsedSentences = sent_detector.tokenize(text.strip())
            for parsedSentence in parsedSentences:
                self.scrappedSentencesList.append(parsedSentence)
                
        if self.scrappedSentencesList:
            retVal = True
        else:
            retVal = False
        return retVal;

    def excludeDisclaimerPartFromScrappedSentences(self, disclaimerStopWords = ['Disclaimer', 'Disclosure']):
        retVal = False        
        newSentenceList = []
        for sentence in self.scrappedSentencesList:
            for disclaimerWord in disclaimerStopWords:
                if disclaimerWord in sentence:
                    retVal = True;
            if retVal:
                break;
            else:
                newSentenceList.append(sentence)
        self.scrappedSentencesList = newSentenceList
        return retVal;

    def excludeAllSentencesContainingStr(self, bannedString):
        retVal = False
        newSentenceList = []
        for sentence in self.scrappedSentencesList:
            retVal = True
            excludeFlag = False
            if bannedString in sentence:
                excludeFlag = True;
            
            if not excludeFlag:
                newSentenceList.append(sentence)
        self.scrappedSentencesList = newSentenceList        
        return retVal;

    def getScrappedSentencesList(self):
        return self.scrappedSentencesList;

    def scrapSentencesInWords(self):
        retVal = False
        self.scrappedWordsList = []        
        
        for originalToken in self.scrappedSentencesList:
            words = originalToken.split(' ')
            for word in words:
                self.scrappedWordsList.append(word)
        if self.scrappedWordsList:
            retVal = True
        else:
            retVal = False
        return retVal;

    def getScrappedWordsList(self):
        return self.scrappedWordsList;

    def scrapSentencesInNamedEntities(self, useStandardNLTKTokenizer = False):
        retVal = 'False'        

        parsedTokens = []
        namedEntities = []
        self.namedEntitiesList = []
        
        for sentence in self.scrappedSentencesList:
            
            if useStandardNLTKTokenizer:
                parsedTokens = nltk.word_tokenize(sentence)
            else:
                tokenizer = RegexpTokenizer(r'\w+')
                parsedTokens = tokenizer.tokenize(sentence)
            
            namedEntities = nltk.pos_tag(parsedTokens)
            for taggedToken in namedEntities:
                self.namedEntitiesList.append(taggedToken)

        if self.namedEntitiesList:
            retVal = True
        else:
            retVal = False
        return retVal;

    def getNamedEntitiesList(self):
         return self.namedEntitiesList;

    def filterNamedEntities(self, exclusionTagList = [NAMED_ENTITY_LABEL_FW, NAMED_ENTITY_LABEL_IN, NAMED_ENTITY_LABEL_JJ, NAMED_ENTITY_LABEL_JJR, NAMED_ENTITY_LABEL_JJS, NAMED_ENTITY_LABEL_MD, NAMED_ENTITY_LABEL_NN, NAMED_ENTITY_LABEL_NNS, NAMED_ENTITY_LABEL_NNP, NAMED_ENTITY_LABEL_NNPS, NAMED_ENTITY_LABEL_RB, NAMED_ENTITY_LABEL_RBR, NAMED_ENTITY_LABEL_RBS, NAMED_ENTITY_LABEL_RP, NAMED_ENTITY_LABEL_SYM, NAMED_ENTITY_LABEL_VB, NAMED_ENTITY_LABEL_VBD, NAMED_ENTITY_LABEL_VBG, NAMED_ENTITY_LABEL_VBP, NAMED_ENTITY_LABEL_VBZ, NAMED_ENTITY_LABEL_WRB], turnLowerCaseFlag = False):
        retVal = False
        self.filteredNamedEntitiesList = []

        for namedEntity in self.namedEntitiesList:
            filterFlag = 0
            for filters in exclusionTagList:
                if namedEntity[1] == filters:
                    filterFlag = filterFlag + 1
                
            if filterFlag==1 and len(namedEntity[0])>1:
                if turnLowerCaseFlag:
                    entityPart = namedEntity[0].lower()
                    posPart = namedEntity[1]
                    self.filteredNamedEntitiesList.append([entityPart, posPart])
                else:
                    self.filteredNamedEntitiesList.append([namedEntity, posPart])

        if self.filteredNamedEntitiesList:
            retVal = True
        else:
            retVal = False
        return retVal;
 
    def getFilteredNamedEntitiesList(self):
        return self.filteredNamedEntitiesList;

    def assignPoSToUnigramms(self, uniGramOccuranciesInText, filteredNamedEntitiesList):
        PoSsedUnigramsList = []
        
        for unigramLine in uniGramOccuranciesInText:
            unigram = unigramLine[0]
            frequency = unigramLine[1]
            
            indexNo = [(k, detectedUnigram.index(unigram))  for k, detectedUnigram in enumerate(filteredNamedEntitiesList) if unigram == detectedUnigram[0]]
            if indexNo:
                posTag = filteredNamedEntitiesList[indexNo[0][0]][1]
                PoSsedUnigramsList.append([unigram, frequency, posTag])
        
        return PoSsedUnigramsList;

    def removeBannedWords(self, posTaggedUnigrams, bannedWordsFullFileName):
        cleanUnigramsList = []
        if not posTaggedUnigrams:
            return cleanUnigramsList;
        
        #Remove Banned Words
        bannedWordsList=[]
        with open(bannedWordsFullFileName) as csvfile:
            parsedCSVFile = csv.reader(csvfile, delimiter=',')
            rowIt = 0    
            for row in parsedCSVFile:
                if rowIt == 0:
                    rowIt = rowIt + 1
                else:
                    if int(row[0]) == 0:
                       bannedWordsList.append(row[1])
                    rowIt = rowIt + 1
        
        if not bannedWordsList:
            return cleanUnigramsList;
        
        for unigramLine in posTaggedUnigrams:
            unigram = unigramLine[0]
            retVal = True
            for bannedWord in bannedWordsList:
                if unigram == bannedWord:
                    retVal = False
            if retVal:
                cleanUnigramsList.append(unigramLine)
        return cleanUnigramsList;

    def wnFilteringPosTaggedUnigrams(self, uniGramOccuranciesWithPoS, printFlag = False):
        stemmer = SnowballStemmer("english")
        wnFilteredUnigramList = []
        #wnPossibleWnSynsetsUsedIndex = [0 for x in range(len(allWordsList))]
        for unigramLine in uniGramOccuranciesWithPoS:
            if printFlag:
                print('New Unigram')
            word1 = unigramLine[0]
            stem1 = stemmer.stem(word1)
            frequency = unigramLine[1]
            pos = unigramLine[2]
            wnPos = self.convertNltkNN2WordnetSynset(pos)
            
            correctSynset = ''
            correctSynsetWord = ''
            if not wnPos == 'NA':
                wnPossibleWnSynsets = wn.synsets(word1, wnPos)
                for wnSynsetIt in range(0, len(wnPossibleWnSynsets)):
                    wnSynset = wnPossibleWnSynsets[wnSynsetIt]
                    synsetWord = wnSynset.lemmas()[0].name()
                    synsetWordStem = stemmer.stem(synsetWord)
                    if synsetWordStem in stem1 or stem1 in synsetWordStem:
                        correctSynsetWord = synsetWord
                        correctSynset = wnSynset
                        wnFilteredUnigramList.append([word1, frequency, pos, correctSynset, correctSynsetWord, stem1])
                        if printFlag:                        
                            print(['OK', word1, pos, wnPos])
                        break
                if printFlag:
                    print(['NOT FOUND', word1, pos, wnPos])
            else:
                if printFlag:
                    print(['ERROR', word1, pos, wnPos])
        return wnFilteredUnigramList;
    
    def appendWnHypernyms(self, wnFilteredUnigramList, synsetColumnID = 3, printFlag = False):
        wnHypernymEnhancedFilteredUnigramList = []
            
        for line in wnFilteredUnigramList:
            synset = line[synsetColumnID]
            hypernyms = synset.hypernyms()
            
            if hypernyms:
                hypernym = hypernyms[0]
            else:
                hypernym = ''
            
            newLine = list(line)
            newLine.append(hypernym)
            if printFlag:
                print(newLine)
            wnHypernymEnhancedFilteredUnigramList.append(newLine)                
            
        return wnHypernymEnhancedFilteredUnigramList;        
         
    def getNoOfWordsInDoc(self):
        return len(self.filteredNamedEntitiesList)
        
#----------------------------------


#------------Put tokens in a text string---------------------- 
    def createStringFromNamedEntities(self, namedEntitiesList, printLabel = False):
        textString = ""
        for namedEntity in namedEntitiesList:
            textString+=str(namedEntity[0])
            textString+=str(" ")
            
            if printLabel:
                print(textString)
        return textString;
        
    def createStringFromTokens(self, tokensList, printLabel = False):
        textString = ""
        for token in tokensList:
            textString+=str(token)
            textString+=str(" ")
            
            if printLabel:
                print(textString)        
        return textString;
#---------------------------------- 

#----------Manage n-Gramms------------------------
    def clearUniqueLabelsList(self):
        self.uniqueITLabelsList = None
        self.uniqueITLabelsList = []
        retVal = True
        return retVal;

    def getUnigramFreqTerms(self, textString, noOfMostCommon = -1, leastFrequencyAllowed = 0, printFlag = False):
        uniGramTokens = nltk.word_tokenize(textString)
        uniGramTokenDistribution = nltk.FreqDist(uniGramTokens)

        if noOfMostCommon == -1:
            noOfMostCommon = len(uniGramTokenDistribution)
        else:
            if len(uniGramTokenDistribution) < noOfMostCommon:
                noOfMostCommon = len(uniGramTokenDistribution)

        mostFrequentWords = uniGramTokenDistribution.most_common(noOfMostCommon)

        mostFrequentWordList = []
        for word in mostFrequentWords:
            if word[1]>leastFrequencyAllowed:
                newRow = [word[0], word[1]]
                mostFrequentWordList.append(newRow)

        if printFlag:
            for word in mostFrequentWordList:
                print(word[0] + ': ' + str(word[1]))

        self.uniGramListWithProbs = mostFrequentWordList;
        return self.uniGramListWithProbs;
    
    def getUniGramFrequencyInDoc(self, unigram):
        retVal = -1
        if self.uniGramListWithProbs:
            indexNo = [(k, detectedUnigram.index(unigram))  for k, detectedUnigram in enumerate(self.uniGramListWithProbs) if unigram == detectedUnigram[0]]
            if indexNo:
                retVal = indexNo[0][0]
        return retVal;
    
    def getBigramsListFromText(self, textString, noOfMostCommon=-1, leastFrequencyAllowed = 0, printFlag = False, windowSize = 2):

        self.biGramListWithProbs = []

        biGramFinder = nltk.collocations.BigramCollocationFinder.from_words(textString.split(' '), window_size = windowSize)    
        biGramFinder.apply_freq_filter(leastFrequencyAllowed)
        #biGramScored = nltk.collocations.BigramAssocMeasures()
    
        biGramScoredList = []
        for k,v in biGramFinder.ngram_fd.items():
            biGramScoredList.append([k, v])

        sortedBiGramScoredList = sorted(biGramScoredList, key=itemgetter(1), reverse=True)
            
        if noOfMostCommon == -1:
            noOfMostCommon = len(sortedBiGramScoredList)
        else:
            if len(sortedBiGramScoredList) < noOfMostCommon:
                noOfMostCommon = len(sortedBiGramScoredList)            
            
        biGramsListWithProbs = []
        for itRow in range(noOfMostCommon):
            biGramWord = str(sortedBiGramScoredList[itRow][0][0]) + " " + str(sortedBiGramScoredList[itRow][0][1])
            tempRow = [biGramWord, sortedBiGramScoredList[itRow][1]]
            biGramsListWithProbs.append(tempRow)            

        self.biGramListWithProbs = biGramsListWithProbs
            
        return self.biGramListWithProbs;    

    def getBigramFrequencyInDoc(self, bigram):
        retVal = -1
        
        indexNo = [(i, detectedBigram.index(bigram))  for i, detectedBigram in enumerate(self.biGramListWithProbs) if bigram == detectedBigram[0]]
        
        retVal = indexNo[0][0]
        
        return retVal;
    
    def getTrigramsListFromText(self, textString, noOfMostCommon=-1, leastFrequencyAllowed = 0, printFlag = False, windowSize = 3):

        self.triGramListWithProbs = []        
        
        triGramFinder = nltk.collocations.TrigramCollocationFinder.from_words(textString.split(' '), window_size = windowSize)    
        triGramFinder.apply_freq_filter(leastFrequencyAllowed)
        #tgm = nltk.collocations.TrigramAssocMeasures()

        triGramScoredList = []
        for k,v in triGramFinder.ngram_fd.items():
            triGramScoredList.append([k, v])

        sortedTriGramScoredList = sorted(triGramScoredList, key=itemgetter(1), reverse=True)

        if noOfMostCommon == -1:
            noOfMostCommon = len(sortedTriGramScoredList)
        else:
            if len(sortedTriGramScoredList) < noOfMostCommon:
                noOfMostCommon = len(sortedTriGramScoredList)

        triGramsListWithProbs = []
        for itRow in range(noOfMostCommon):
            triGramWord = str(sortedTriGramScoredList[itRow][0][0]) + " " + str(sortedTriGramScoredList[itRow][0][1]) + " " + sortedTriGramScoredList[itRow][0][2]
            tempRow = [triGramWord, sortedTriGramScoredList[itRow][1]]
            triGramsListWithProbs.append(tempRow) 

        self.triGramListWithProbs = triGramsListWithProbs

        return self.triGramListWithProbs;
       
    def getTrigramFrequencyInDoc(self, trigram):
        retVal = -1
        
        indexNo = [(i, detectedTrigram.index(trigram))  for i, detectedTrigram in enumerate(self.triGramListWithProbs) if trigram == detectedTrigram[0]]
        
        retVal = indexNo[0][0]
        
        return retVal;

    def removeOverlappings(self, unigrams, bigrams, trigrams, warningPrintFlag = False, debugMonitoringFlag = False):
        
        self.nonOverlappingTokensFoundFlag = False;        

        self.cleanUnigrams = []
        self.cleanBigrams = []
        self.cleanTrigrams = []
        self.cleanTrigrams = trigrams
                
        #remove Bigramms from Trigramms
        for bigram in bigrams:
            excludeCounter = 0
            for trigram in trigrams:
                if self.unsplitStringContainedInString(bigram[0], trigram[0]):
                    excludeCounter = excludeCounter + 1
            if excludeCounter>0:
                if (bigram[1] - excludeCounter > 0):
                    if debugMonitoringFlag:
                        print('DEBUG NOTE:Decreasing Bigram: ' + bigram[0] + ', +' + str(bigram[1]) + ', -' + str(excludeCounter))
                    newRow = [bigram[0], bigram[1] - excludeCounter]
                    self.cleanBigrams.append(newRow)
                elif (bigram[1] - excludeCounter < 0) and warningPrintFlag:
                    print('WARNING: Trigram appears more often than Bigram: ' + bigram[0])
            else:
                if debugMonitoringFlag:
                    print('DEBUG NOTE:Preserving Bigram: ' + bigram[0] + ', +' + str(bigram[1]) + ', -' + str(excludeCounter))
                newRow = [bigram[0], bigram[1]]
                self.cleanBigrams.append(newRow)

        #remove Unigramms from Bigrams and Trigramms
        for unigram in unigrams:
            excludeCounter = 0
            for bigram in bigrams:
                if self.unsplitStringContainedInString(unigram[0], bigram[0]):
                    excludeCounter = excludeCounter + 1
            if excludeCounter>0:
                if (unigram[1] - excludeCounter > 0):
                    if debugMonitoringFlag:
                        print('DEBUG NOTE:Preserving Unigram: ' + unigram[0] + ', +' + str(unigram[1]) + ', -' + str(excludeCounter))
                    newRow = [unigram[0], unigram[1] - excludeCounter]
                    self.cleanUnigrams.append(newRow)
                elif (unigram[1] - excludeCounter < 0) and warningPrintFlag:
                    print('WARNING: Bigram appears more often than Unigram: ' + unigram[0])
            else:
                if debugMonitoringFlag:
                    print('DEBUG NOTE:Preserving Unigram: ' + unigram[0] + ', +' + str(unigram[1]) + ', -' + str(excludeCounter))
                newRow = [unigram[0], unigram[1]]
                self.cleanUnigrams.append(newRow)

        self.nonOverlappingTokensFoundFlag = True
        
        return self.nonOverlappingTokensFoundFlag;
    
    def unsplitStringContainedInString(self, unsplitString, string):
        retVal = False
        stringList = string.split()

        for word in stringList:
            if unsplitString == word:
                retVal = True
                
        return retVal;

    def getCleanUnigrams(self):
        return self.cleanUnigrams;
    
    def getCleanBigrams(self):
        return self.cleanBigrams;
        
    def getCleanTrigrams(self):
        return self.cleanTrigrams;
        
#----------------------------------       

#----------Manage (Wordnet) Hypernym Hierarchy------------------------    
    def createHypernymsList(self, sunsetFlag = False, printFlag = False):

        self.hypernymHierarchyList = []
        if not self.filteredNamedEntitiesList and not self.uniGramListWithProbs:
            return self.hypernymHierarchyList;
        
        synsets = []
        entities = []
        hierarchies = []
        frequencies = []
        i = 0
        bannedSynset = self.getActiveSynset('synset')
        for unigramWithProb in self.uniGramListWithProbs:
            i = i + 1
            j = 0
            for unigramWithPosTag in self.filteredNamedEntitiesList:   
                j = j + 1             
                if unigramWithProb[0] == unigramWithPosTag[0]:
        
                    unigramName = unigramWithPosTag[0]
                    frequency = unigramWithProb[1]
                    indexNo = [(k, detectedUnigram.index(unigramName))  for k, detectedUnigram in enumerate(self.filteredNamedEntitiesList) if unigramName == detectedUnigram[0]]
                    posTag =  self.filteredNamedEntitiesList[indexNo[0][0]][1]
                    
                    synset = self.getActiveSynset(unigramName, posTag)
                    if not (bannedSynset == synset):
                        hierarchy1 = self.getContextualHypernym(synset)
                        if hierarchy1 and not (bannedSynset == synset):
                            entities.append(unigramName)                            
                            synsets.append(synset)                            
                            hierarchies.append(hierarchy1)
                            frequencies.append(frequency)
                        else:
                            synset = self.getActiveSynset(unigramName)
                            if not (bannedSynset == synset):
                                hierarchy2 = self.getContextualHypernym(synset)
                                if hierarchy2:
                                    entities.append(unigramName)
                                    synsets.append(synset)
                                    hierarchies.append(hierarchy2)
                                    frequencies.append(frequency)
                                else:
                                    if printFlag:
                                        print('!!!!WARNING: ' + unigramName)
                    break
            if printFlag:
                print(str(i) + ': ' + unigramWithProb[0] + ' ' + str(j) + ': ' + unigramWithPosTag[0])
        
        minLen = 1000
        maxLen = 0
        for hierarchy in hierarchies:
            if maxLen<len(hierarchy):
               maxLen=len(hierarchy)
            if minLen>len(hierarchy):
                minLen=len(hierarchy)
                
        for hierarchy in hierarchies:
            indexLast = len(hierarchy)
            diff = maxLen - indexLast
            for i in range(diff):
                hierarchy.append(hierarchy[indexLast-1])
        
        if (len(hierarchies) == len(entities)) and (len(hierarchies) == len(frequencies) and len(hierarchies) == len(synsets)):
            i = 0
            for line in hierarchies:
                fullMatrixLine = []
                fullMatrixLine.append(entities[i])
                for word in line:
                    if sunsetFlag:
                        fullMatrixLine.append(word)
                    else:
                        fullMatrixLine.append(word.lemmas()[0].name())
                if sunsetFlag:
                    fullMatrixLine.append(synsets[i])
                else:
                    fullMatrixLine.append(synsets[i].lemmas()[0].name())
                fullMatrixLine.append(frequencies[i])
                i = i + 1
                self.hypernymHierarchyList.append(fullMatrixLine)

        return self.hypernymHierarchyList;

    def createHypernymsListWithoutBannedWord(self, bannedWordsFullFileName, sunsetFlag = False, printFlag = False):

        self.hypernymHierarchyList = []
        if not self.filteredNamedEntitiesList and not self.uniGramListWithProbs:
            return self.hypernymHierarchyList;
        
        #Remove Banned Words
        bannedWordsList=[]
        with open(bannedWordsFullFileName) as csvfile:
            parsedCSVFile = csv.reader(csvfile, delimiter=',')
            rowIt = 0    
            for row in parsedCSVFile:
                if rowIt == 0:
                    rowIt = rowIt + 1
                else:
                    if int(row[0]) == 0:
                       bannedWordsList.append(row[1])
                    rowIt = rowIt + 1
        
        filteredInformationGainList = []
        for rowIt in range(0, len(self.filteredNamedEntitiesList)):
            IGWord = self.filteredNamedEntitiesList[rowIt][0]
            IGValue = self.filteredNamedEntitiesList[rowIt][1]
            indexNo = 0
            try:
                indexNo = bannedWordsList.index(IGWord)
            except ValueError:  
                indexNo = -1
            if indexNo == -1:
                filteredInformationGainList.append([IGWord, IGValue])
            
        self.filteredNamedEntitiesList = filteredInformationGainList

        
        synsets = []
        entities = []
        hierarchies = []
        frequencies = []
        i = 0
        bannedSynset = self.getActiveSynset('synset')
        for unigramWithProb in self.uniGramListWithProbs:
            i = i + 1
            j = 0
            for unigramWithPosTag in self.filteredNamedEntitiesList:   
                j = j + 1             
                if unigramWithProb[0] == unigramWithPosTag[0]:
        
                    unigramName = unigramWithPosTag[0]
                    frequency = unigramWithProb[1]
                    indexNo = [(k, detectedUnigram.index(unigramName))  for k, detectedUnigram in enumerate(self.filteredNamedEntitiesList) if unigramName == detectedUnigram[0]]
                    posTag =  self.filteredNamedEntitiesList[indexNo[0][0]][1]
                    
                    synset = self.getActiveSynset(unigramName, posTag)
                    if not (bannedSynset == synset):
                        hierarchy1 = self.getContextualHypernym(synset)
                        if hierarchy1 and not (bannedSynset == synset):
                            entities.append(unigramName)                            
                            synsets.append(synset)                            
                            hierarchies.append(hierarchy1)
                            frequencies.append(frequency)
                        else:
                            synset = self.getActiveSynset(unigramName)
                            if not (bannedSynset == synset):
                                hierarchy2 = self.getContextualHypernym(synset)
                                if hierarchy2:
                                    entities.append(unigramName)
                                    synsets.append(synset)
                                    hierarchies.append(hierarchy2)
                                    frequencies.append(frequency)
                                else:
                                    if printFlag:
                                        print('!!!!WARNING: ' + unigramName)
                    break
            if printFlag:
                print(str(i) + ': ' + unigramWithProb[0] + ' ' + str(j) + ': ' + unigramWithPosTag[0])
        
        minLen = 1000
        maxLen = 0
        for hierarchy in hierarchies:
            if maxLen<len(hierarchy):
               maxLen=len(hierarchy)
            if minLen>len(hierarchy):
                minLen=len(hierarchy)
                
        for hierarchy in hierarchies:
            indexLast = len(hierarchy)
            diff = maxLen - indexLast
            for i in range(diff):
                hierarchy.append(hierarchy[indexLast-1])
        
        if (len(hierarchies) == len(entities)) and (len(hierarchies) == len(frequencies) and len(hierarchies) == len(synsets)):
            i = 0
            for line in hierarchies:
                fullMatrixLine = []
                fullMatrixLine.append(entities[i])
                for word in line:
                    if sunsetFlag:
                        fullMatrixLine.append(word)
                    else:
                        fullMatrixLine.append(word.lemmas()[0].name())
                if sunsetFlag:
                    fullMatrixLine.append(synsets[i])
                else:
                    fullMatrixLine.append(synsets[i].lemmas()[0].name())
                fullMatrixLine.append(frequencies[i])
                i = i + 1
                self.hypernymHierarchyList.append(fullMatrixLine)

        return self.hypernymHierarchyList;

    def getMaxExistingHierarchyLayer(self):

        retVal = -1
        
        if(self.hypernymHierarchyList):
            retVal = len(self.hypernymHierarchyList[0]) - 3
        
        return retVal;
        
    def retrieveCertainLayerFromHierarchy(self, hierarchyLayer):

        retVal = []        
        if len(self.hypernymHierarchyList) > 0:
            if(hierarchyLayer > len(self.hypernymHierarchyList[0]) - 3):
                return retVal;
    
            if(self.hypernymHierarchyList):
                i = 0
                for line in self.hypernymHierarchyList:
                    j=0
                    hierarchyLine = []
                    for word in line:
                        #if j==0:
                            #hierarchyLine.append(word)
                        if hierarchyLayer == j-1:
                            hierarchyLine.append(word)
                        elif j==len(line)-1:
                            hierarchyLine.append(int(word))
    
                        j = j + 1
                    retVal.append(hierarchyLine)
                    i = i + 1
                    pass

        return retVal;

    def getActiveSynset(self, entity, nltkPosTag = ''):

        synset = wn.synsets('synset')[0]
        
        #lematize
        lmtzr = WordNetLemmatizer()
        lemma = lmtzr.lemmatize(entity)
        
        synsets = []
        if nltkPosTag:
            wnPosTag = self.convertNltkNN2WordnetSynset(nltkPosTag)
            if wnPosTag == 'NA':
                synsets = wn.synsets(lemma)
            else:
                synsets = wn.synsets(lemma, wnPosTag)
        else:
            synsets = wn.synsets(lemma)
        
        if synsets:
            #get hypernyms
            k = 0
            hypernyms = []
            while not hypernyms and k<len(synsets):
                synset = synsets[k]
                k = k + 1
                hypernyms = synset.hypernyms()

        return synset;

    def getContextualHypernym(self, synset, printHierarchyFlag = False):     
        
        stemmingTree = []
        hypernyms = synset.hypernyms()
        if hypernyms:
            hypernym = hypernyms[0]
            iCount = 1
            stemmingTree.append(hypernym)
            while hypernym:
                nextSetOfHypernyms = hypernym.hypernyms()
                if nextSetOfHypernyms:
                    hypernym = nextSetOfHypernyms[0]
                    iCount = iCount + 1
                    stemmingTree.append(hypernym)
                else:
                    hypernym = ''
        
        stemmingTree.reverse()
        if printHierarchyFlag:
            print(stemmingTree)
        
        retVal = []
        if stemmingTree:
            for hypernym in stemmingTree:
                retVal.append(hypernym)
        else:
            retVal = ''
        
        return retVal;
#----------------------------------    

    def condenseListByCertainColumn(self, entitiesList, condenseBasedOnColumnID = 0, printFlag = False):

        NoW = -1        
        
        condensedList = []        
        if entitiesList:
            return condensedList;
        
        uniqueItemList = []
        for row in entitiesList:
            uniqueItemList.append(row[condenseBasedOnColumnID])
            
        uniqueItemList = list(set(uniqueItemList))
        
        NoW = 0
        for uniqueItem in uniqueItemList:
            aggregatedScore = 0
            uniqueItem = uniqueItemList
            for entitiesListLine in entitiesList:
                if uniqueItem == entitiesListLine[condenseBasedOnColumnID]:                    
                    word1 = entitiesListLine[0]
                    frequency = entitiesListLine[1]
                    pos = entitiesListLine[2]
                    correctSynset = entitiesListLine[3]
                    correctSynsetWord = entitiesListLine[4]
                    stem1 = entitiesListLine[5]
                    hypernym1 = entitiesListLine[6]
                    
                    NoW = NoW + frequency
                    aggregatedScore = aggregatedScore + frequency

                    if printFlag:
                        print(uniqueItem + ' ' + ' ' + str(aggregatedScore))
                        
                    condensedList.append([word1, aggregatedScore, pos, correctSynset, correctSynsetWord, stem1, hypernym1])
        
        condensedList = sorted(condensedList, key=itemgetter(1), reverse=True)
            
        return condensedList;     

    def condenseHypernymList(self, hypernymsList, condenseBasedOnColumnID = 0, printFlag = False):
        
        self.NoW = -1        
        
        self.uniqueSynsetListWithScores = []
        if len(hypernymsList) == 0:
            return self.uniqueSynsetListWithScores;
        
        uniqueSynsetList = []
        for row in hypernymsList:
            uniqueSynsetList.append(row[0])
            
        uniqueSynsetList = list(set(uniqueSynsetList))
        
        self.NoW = 0
        for uniqueSynset in uniqueSynsetList:
            aggregatedScore = 0
            for hypernymWithScore in hypernymsList:
                if uniqueSynset == hypernymWithScore[0]:
                    self.NoW = self.NoW + hypernymWithScore[1]
                    aggregatedScore = aggregatedScore + hypernymWithScore[1]
                    if printFlag:
                        print(uniqueSynset + ' ' + ' ' + str(aggregatedScore))
                    extraLine = []
                    for itemIt in range(2, len(hypernymWithScore)):
                        extraLine.append(hypernymWithScore[itemIt])

            self.uniqueSynsetListWithScores.append([uniqueSynset, int(aggregatedScore)] + extraLine)                        
                
        self.uniqueSynsetListWithScores = sorted(self.uniqueSynsetListWithScores, key=itemgetter(1), reverse=True)
            
        return self.uniqueSynsetListWithScores;
    
    def getNoWInDoc(self, inputList, frequencyColumnID = 1):

        NoW = 0

        for line in inputList:
            NoW = NoW + int(line[frequencyColumnID])
        
        return NoW;
        
    def getNoOfConceptualEntitiesInDoc(self):
        return self.NoW;
        
    def getDocEntityProbabilityList(self, synsetFlag = False):

        self.entityProbabilityInText = []
        if self.uniqueSynsetListWithScores and self.NoW>0:
            for entityLine in self.uniqueSynsetListWithScores:
                if synsetFlag:
                    entity = entityLine[0].lemmas()[0].name()
                else:
                    entity = entityLine[0]
                entityFrequency = entityLine[1]
                self.entityProbabilityInText.append([entity, float(entityFrequency/self.NoW)])
                
        return self.entityProbabilityInText;
