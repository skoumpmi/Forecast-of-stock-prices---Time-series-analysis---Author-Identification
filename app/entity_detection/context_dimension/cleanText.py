import string
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def removeNonAlphabeticSymbolsFromString(originalTextString, excludedSymbolsList = [64, 65, 66, 67, 71, 72, 76, 77, 78, 79, 80, 81, 83, 90, 91, 92, 93]): ##$%&*+/:;=^{|}
    '''
    string.printable = 
    '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
    '''

    onlyAlphabeticTokensList = []

    for i in excludedSymbolsList:
        cleanToken = originalTextString.replace(string.printable[i], "");
        onlyAlphabeticTokensList.append(cleanToken);

    return onlyAlphabeticTokensList;
    
def filterTokensFromTokensList(originalTokensList, bannedTokensList):

    filteredTokenList = []
    exclusionFlag = False
    for originalToken in originalTokensList:
        exclusionFlag = False
        for bannedToken in bannedTokensList:
            if (((originalToken in bannedToken) or (bannedToken in originalToken)) or ((originalToken.upper() in bannedToken.upper()) or (bannedToken.upper() in originalToken.upper()))):
                exclusionFlag = True
            else:
                exclusionFlag = False

        if not exclusionFlag:
            filteredTokenList.append(originalToken)

    return filteredTokenList;

def filterTokensFromNamedEntitiesList(originalNamedEntitiesList, bannedTokensList):

    filteredTokenList = []
    exclusionFlag = False
    for originalNamedEntity in originalNamedEntitiesList:
        exclusionFlag = False
        for bannedToken in bannedTokensList:
            if (((originalNamedEntity[0] in bannedToken) or (bannedToken in originalNamedEntity[0])) or ((originalNamedEntity[0].upper() in bannedToken.upper()) or (bannedToken.upper() in originalNamedEntity[0].upper()))):
                exclusionFlag = True
            else:
                exclusionFlag = False

        if not exclusionFlag:
            filteredTokenList.append(originalNamedEntity)

    return filteredTokenList;

def stopAtDisclamer(originalTokenList, listOfDisWords):
    
    filteredTokenList = []
    continueFlag = True
    iterator = -1;
    while (continueFlag and (iterator <= len(originalTokenList)-2)):
        iterator = iterator + 1;
        filteredTokenList.append(originalTokenList[iterator])
        for disWord in listOfDisWords:
            if disWord in originalTokenList[iterator][0]:
                continueFlag = False
    
    return filteredTokenList;
    
def stopAtDisclamerforSentenceList(originalSentenceList, listOfDisWords):
    
    filteredSentenceList = []
    continueFlag = True
    iterator = -1;
    while continueFlag and (iterator <= len(originalSentenceList)-2):
        iterator = iterator + 1;
        filteredSentenceList.append(originalSentenceList[iterator])
        for disWord in listOfDisWords:
            if disWord in originalSentenceList[iterator]:
                continueFlag = False
    
    return filteredSentenceList;
    
def startAtDisclamer(originalTokenList, listOfDisWords):
    
    filteredTokenList = []
    startFlag = False
    iterator = -1;
    while (iterator <= len(originalTokenList)-2):
        iterator = iterator + 1;
        if not startFlag:
            for disWord in listOfDisWords:
                if disWord in originalTokenList[iterator][0]:
                    startFlag = True
        else:
            filteredTokenList.append(originalTokenList[iterator])
    
    return filteredTokenList;

def textPreprocessing(text, lowercase=True, removePunctuation=False, removeNonAlphabetic=False, 
                      removeStopWords=True, splitHashTags=False):
        text = text.replace("b'","")
        #------hashTags------#
        if splitHashTags:
            hashtags = re.findall(r'#\b.+?\b', text)
            new_hashtags = []        
            for hashtag in hashtags:
                new_hashtag_list = re.findall('[a-zA-Z][^A-Z1-9]{2,}|[A-Z]{2,}', hashtag)
                new_hashtag = ' '.join(new_hashtag_list)
                new_hashtags.append(new_hashtag)
                text = text.replace(hashtag, new_hashtag)            
                
        if lowercase:
            text = text.lower()
        
        tokens = word_tokenize(text)  

        if removePunctuation:
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            tokens = [w.translate(table) for w in tokens]
        if removeNonAlphabetic:
            # remove remaining tokens that are not alphabetic
            tokens = [word for word in tokens if word.isalpha()]
        if removeStopWords:   
            # filter out stop words
            tokens = [w for w in tokens if not w in stop_words]
        new_text = ' '.join(tokens)
        #remove stopWords
        return new_text