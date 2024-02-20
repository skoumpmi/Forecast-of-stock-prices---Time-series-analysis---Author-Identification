
import configparser
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from sklearn.model_selection import train_test_split
import string
from collections import Counter
from scipy import stats
import re
import os
from ..database.dbManager import DBManager


class ReliabilityCalculator:
    def __init__(self):
        self.db_manager = DBManager()

    def clean_text(self, text):
        text = re.sub('[^.,\.!?a-zA-Z0-9 \n\.]', '', text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        # text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def create_train_dataset(self, data, group_feature, new_condition_1, profile_length):
        data = data.copy()
        if data[(data[new_condition_1] == 0) & (data[group_feature].notna())].reset_index(drop=True).empty:
            new_data = data.copy()
        else:
            new_data = data[(data[new_condition_1] == 0) & (data[group_feature].notna())].reset_index(drop=True)
        uniqueVals = new_data[group_feature].unique()
        grouped = new_data.groupby(group_feature)
        dat = pd.concat([pd.DataFrame([i], columns=[group_feature]) for i in uniqueVals], ignore_index=True)
        dat['full_text'] = ''
        for i, row in dat.iterrows():
            dat.loc[i, 'full_text'] = ' '.join(grouped.get_group(uniqueVals[i])['full_text'])
            dat.loc[i, 'full_text'] = self.clean_text(dat.loc[i, 'full_text'])
        dat['text_list'] = 0
        dat['chunks'] = 0
        all = ''
        processed_items = []
        for i in range(0, len(dat)):
            if len(dat['full_text'][i].split()) >= profile_length:
                dat['text_list'][i] = ((dat['full_text'][i].split()[:profile_length]))
                dat['chunks'][i] = (
                    list(map(" ".join, zip(*[iter(dat['text_list'][i])] * int(profile_length / 10)))))
                processed_items.append(dat[group_feature][i])
                all += ' '.join(dat['full_text'][i].split()[: profile_length])
        for i, row in data.iterrows():
            if data.loc[i, group_feature] in processed_items:
                data.loc[i, new_condition_1] = 1
        dat = (dat[dat['chunks'] != 0].reset_index(drop=True))  # NEA GRAMMI
        dat['train_text'] = 0
        for j in range(0, len(dat)):
            X_train, X_test = train_test_split(dat['chunks'][j], test_size=(0.1), random_state=42, shuffle=True)
            dat['train_text'][j] = X_train
        dat = dat.loc[:, [group_feature,'train_text']]

        return dat, data, all

    def find_most_frequent_function_words(self, data, group_feature,
                                                  new_condition_1,
                                                  profile_length, nb_of_function, all_text_concatenated):
                words = ['a', 'bit', 'couple', 'aboard', 'about', 'above', 'absent', 'according',
                'accordingly', 'across', 'after', 'against', 'ahead',
                'albeit', 'all', 'along', 'alongside', 'although', 'amid', 'amidst', 'among', 'amongst',
                'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'around', 'as' ,
                'aside', 'astraddle', 'astride','at', 'away', 'bar', 'barring', 'because', 'before',
                'behind', 'below', 'beneath', 'beside', 'besides','between', 'beyond', 'both', 'but', 'by',
                'can', 'certain', 'circa', 'close', 'concerning', 'consequently',
                'considering', 'could', 'dare', 'despite', 'down', 'due', 'during', 'each', 'either', 'enough',
                'every', 'everybody', 'everyone', 'everything', 'except', 'excluding', 'failing', 'few', 'fewer', 'following',
                'for', 'from', 'given', 'heaps', 'hence', 'however', 'if', 'in', 'spite', 'view', 'including', 'inside',
                'instead', 'into', 'it', 'its', 'itself', 'less', 'like', 'little', 'loads', 'lots',
                'many', 'may', 'might', 'minus', 'more', 'most', 'much', 'must',
                'near', 'need', 'neither', 'nevertheless', 'next', 'no', 'nobody', 'none', 'nor',
                'nothing', 'notwithstanding', 'of', 'off', 'on', 'once', 'one', 'onto', 'opposite', 'or',
                'other', 'ought', 'our', 'out','outside', 'over', 'part', 'past', 'pending', 'per',
                'pertaining', 'plenty', 'plus', 'regarding', 'respecting', 'round', 'save', 'saving', 'several', 'shall', 'should',
                'similar', 'since', 'so', 'some', 'somebody', 'something', 'such', 'than', 'that', 'the', 'them', 'themselves',
                'then', 'thence', 'therefore', 'these', 'they', 'this', 'tho', 'those', 'though', 'through',
                'throughout', 'thru', 'till', 'to', 'toward', 'towards', 'under',
                'underneath', 'unless', 'unlike', 'until', 'unto', 'up', 'upon', 'us', 'used', 'various', 'versus',
                'via', 'wanting', 'what', 'whatever', 'when', 'whenever', 'where', 'whereas', 'wherever', 'whether',
                'which', 'whichever', 'while', 'whilst', 'who', 'whoever', 'whom', 'whomever',
                'whose', 'will', 'with', 'within', 'without', 'would', 'yet']
                function_words = pd.DataFrame(data=words, columns=['{}'.format('function_words')])
                function_words['count'] = 0
                concated_text = all_text_concatenated.split()
                counts = dict(Counter(concated_text))
                for i in range(0, len(function_words)):
                    for key, value in counts.items():
                        if key == function_words['function_words'][i]:
                            function_words['count'][i] = value
                function_words = function_words[function_words['count'] != 0].reset_index(drop=True)
                function_words = function_words.sort_values(by=['count'], ascending=False).reset_index(drop=True)[:nb_of_function]

                return function_words

    def KL_divergence(self, p, q):
        kl = p * np.log(p / q)
        kl = np.where((kl == np.inf), 0, kl)
        kl = np.where((kl == -np.inf), 0, kl)
        kl = np.nan_to_num(kl, nan=0)
        kl = np.sum(kl)
        return kl

    def find_credibility(self, data, group_feature, new_condition_1, profile_length, nb_of_function):
        n = 10

        train_data, data, all_text_concatenated = self.create_train_dataset(data, group_feature, new_condition_1, profile_length)
        # I function mporei na min periexei 50 lekseis opote auto dimiourgei provlima stin KL_divergence opou thelei na einai idiou megethous oi 50x50 pinakes train kai test
        function = self.find_most_frequent_function_words( data, group_feature, new_condition_1, profile_length, nb_of_function, all_text_concatenated)
        counter = 0
        items_score = []
        items_name = []
        id_score = []
        code_path = os.getcwd()
        initial_path = os.path.join(os.getcwd(), "app", "reliability")
        for q in range(0, len(train_data)):
            frame = pd.DataFrame(index=function['function_words'], columns=function['function_words'])
            for i in function['function_words']:
                for j in function['function_words']:
                    frame[i][j] = round(0, 0)
            for i in range(0, 9):
                mytext = train_data['train_text'][q][i]  # [i]#[1]
                for f in range(0, len(sent_tokenize(mytext))):
                    k = (sent_tokenize(mytext)[f].translate(
                        str.maketrans('', '', string.punctuation)).lower().split())
                    for l in range(0, n):
                        k.append('')
                    for j in range(0, len(k)):
                        for m in function['function_words']:
                            if m == k[j]:
                                for z in range(1, n + 1):
                                    for y in function['function_words']:
                                        if (y == (k[j + z])):
                                            frame[y][m] += 0.75 ** (z - 1)
            frame['sum'] = 0
            frame['sum'] = frame.sum(axis=1)
            for k in range(0, len(frame)):
                for l in function['function_words']:
                    frame[l][k] = (round((frame[l][k]) / frame['sum'][k], 3))  # ((frame[l][k]) / frame[sum_feature][k])
            frame = frame.fillna(value=(1 / len(frame)))

            try:
                os.mkdir('%s/%s' % (initial_path, '{}'.format(group_feature)))
                os.chdir('%s/%s' % (initial_path, '{}'.format(group_feature)))
            except OSError:
                os.chdir('%s/%s' % (initial_path, '{}'.format(group_feature)))
                frame.iloc[:, :-1].to_csv('{}.csv'.format(re.sub('[^A-Za-z0-9]+', ' ', str(train_data[group_feature][q]))))
                # items_score.append(str(train_data[group_feature][q]))
                id_score.append((train_data[group_feature][q]))
            # items_name = [list(x) for x in zip(items_score, id_score)]
            os.chdir(code_path)
        os.chdir('%s/%s' % (initial_path, '{}'.format(group_feature)))
        entries1 = os.listdir(os.getcwd())
        score_list = []
        for i in range(0, len(entries1)):
            entries1[i] = entries1[i][:-4]
        if len(entries1) > 1:
            each_item_index = 0
            for item in entries1:
                each_item_index += 1
                print (str(each_item_index) + "/" + str(len(entries1)) + " items: Calculating reliablility for " + group_feature + " " + str(item))
                previous = next_ = None
                l = len(entries1)
                for index, obj in enumerate(entries1):
                    if obj == item:
                        if index > 0:
                            previous = entries1[:index]
                        if index < (l - 1):
                            next_ = entries1[index + 1:]
                if next_ == None:
                    next_ = []
                elif previous == None:
                    previous = []
                final_list = previous + next_
                train = pd.read_csv('{}.csv'.format(item), index_col=0)
                train = train.values.flatten()
                score = 0
                score_list_new = []
                for author in final_list:
                    test = pd.read_csv('{}.csv'.format(author), index_col=0)
                    test = test.values.flatten()
                    kl = self.KL_divergence(train, test)
                    score += kl
                # score = 15756.01605474139
                score_list.append(1 / (round(((score / len(final_list))), 2)))
                os.chdir(code_path)
                os.chdir('%s/%s' % (initial_path, '{}'.format(group_feature)))
                score = 0
            score_list_new = []
            entries1 = os.listdir(os.getcwd())
            name_list = []
            credibility_score = []
            for z in range(0, len(entries1)):
                entries1[z] = entries1[z][:-4]
                name_list.append(entries1[z])
                credibility_score.append(round(((98 * score_list[z]) / max(score_list)), 2))

            my_new_list = [list(x) for x in zip(id_score, credibility_score)]

            for item in my_new_list:
                if item[1] > 50.0:
                    item[1] = round((item[1] + 0.5 * (98 - item[1])), 2)
                else:
                    item[1] = round((item[1] + 0.6 * (98 - item[1])), 2)
            credibility_frame = pd.DataFrame(data=my_new_list, columns=['{}'.format(group_feature),'score'])
        else:
            my_new_list = [[entries1[0], data['{}'.format(group_feature)][0],98]]
            credibility_frame = pd.DataFrame(data=my_new_list, columns=['{}'.format(group_feature),'score'])
        os.chdir(code_path)
        return credibility_frame, data

    def calculate_writer_reliability(self):
        full_text_per_writer = self.db_manager.get_writer_full_text()
        if len(full_text_per_writer) > 0:
            credibility_frame_writer, writer_df = self.find_credibility(full_text_per_writer , 'writer_id','is_processed', 1100, 50)
            return credibility_frame_writer, writer_df
        else:
            column_names = ["a", "b", "c"]
            emty_df = pd.DataFrame(columns = column_names)
            return emty_df, emty_df

    def calculate_source_reliability(self):
        full_text_per_source = self.db_manager.get_source_full_text()
        if len(full_text_per_source) > 0:
            credibility_frame_source, source_df = self.find_credibility(full_text_per_source, 'source_id' ,'is_processed', 1100, 50)
            return credibility_frame_source, source_df
        else:
            column_names = ["a", "b", "c"]
            emty_df = pd.DataFrame(columns = column_names)
            return emty_df, emty_df

    def update_source_and_writer_reliability(self):
        try:
            writer_cred, writer_df = self.calculate_writer_reliability()
            if writer_cred.empty == False and writer_df.empty == False:
                self.db_manager.update_writers_reliability(writer_cred)
                self.db_manager.update_is_processed(writer_df, "writer")
            else:
                print("No Data to update writer reliability!")
            
            source_cred, source_df = self.calculate_source_reliability()
            if source_cred.empty == False and source_df.empty == False:
                self.db_manager.update_source_reliability(source_cred)
                self.db_manager.update_is_processed(source_df, "source")
            else:
                print("No Data to update source reliability!")

        except Exception as e:
            print(str(e))
        
