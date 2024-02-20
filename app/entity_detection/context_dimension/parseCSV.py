# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:02:01 2018

@author: zamihos
"""

import csv
     
def readCSV(fileName):
    retTweets = []

    with open(fileName, newline='') as csvfile:
        tweetsReader = csv.reader(csvfile)            
        for row in tweetsReader:
            retTweets.append(row)
    csvfile.close()
        
    return retTweets;

def readCSVByKeyword(fileName, keyword):
    retTweets = []

    keyword = keyword.lower()
    with open(fileName, newline='') as csvfile:
        tweetsReader = csv.reader(csvfile)            
        for row in tweetsReader:
            if keyword == row[2].lower()[1:]:
                retTweets.append(row)
    csvfile.close()
        
    return retTweets;
        