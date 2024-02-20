import re
import csv
import glob
import random
import urllib
import string
import os, io
import tarfile
import zipfile
import fnmatch
from torchtext import data
import multiprocessing as mp
from torchtext import datasets
from tqdm.auto import tqdm, trange
tqdm.pandas(desc='Progress', position=0, leave=True)

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.
    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root, task):

        if not os.path.exists(os.path.join(root, task)):
            os.makedirs(os.path.join(root, task))

        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, task, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            _, file_extension = os.path.splitext(tpath)
            if not tpath.endswith('.zip'):
                with tarfile.open(tpath, 'r') as tfile:
                    for item in tfile.getmembers():
                        if item.isreg():  # skip if the TarInfo is not files
                            item.name = os.path.basename(item.name)
                            tfile.extract(item, os.path.join(root, task))
                    # tfile.extractall(root)
            else:
                with zipfile.ZipFile(tpath, 'r') as zip_ref:
                    for name in zip_ref.namelist():
                        zip_ref.extract(name, os.path.join(root, task))
                    #zip_ref.extractall(root)
        return os.path.join(root, task)


class MR(TarDataset):
    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    dirname = 'yelp'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, preprocess_txt=False, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        def clean_test_mr(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.lower().strip().split(' ')  # string.strip()

        if preprocess_txt:
            text_field.preprocessing = clean_test_mr
        #else:
        #	text_field.preprocessing = lambda s: s.split()

        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'rt-polarity.neg'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'rt-polarity.pos'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.data', **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root, 'MR')
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio * len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))

class YELP(TarDataset):
    url = 'https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz'
    filename = 'yelp_review_polarity_csv.tar'
    dirname = 'yelp_review_polarity_csv'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, preprocess_txt, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        def clean_text_yelp(text):
            # https://github.com/msahamed/yelp_comments_classification_nlp/blob/master/word_embeddings.ipynb
            ## Remove puncuation
            text =  text.translate(string.punctuation)

            ## Convert words to lower case and split them
            text = text.lower().split()

            ## Remove stop words
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops and len(w) >= 3]

            text = " ".join(text)

            ## Clean the text
            text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
            text = re.sub(r"what's", "what is ", text)
            text = re.sub(r"\'s", " ", text)
            text = re.sub(r"\'ve", " have ", text)
            text = re.sub(r"n't", " not ", text)
            text = re.sub(r"i'm", "i am ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            text = re.sub(r",", " ", text)
            text = re.sub(r"\.", " ", text)
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

            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            #text = " ".join(stemmed_words)

            return stemmed_words

        if preprocess_txt:
            text_field.preprocessing = clean_text_yelp
        #else:
        #	text_field.preprocessing = lambda s: s.split()

        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = {}
            train_examples = []
            test_examples = []
            with open(os.path.join(path, 'train.csv'), errors='ignore') as f:
                row_count_train = sum(1 for row in csv.reader(f, delimiter=','))

            with open(os.path.join(path, 'train.csv'), errors='ignore') as f:
                csv_reader = csv.reader(f, delimiter=',')

                train_examples=[]
                for line in tqdm(csv_reader, total=row_count_train, desc="Loading train data", leave=True, position=0):
                    train_examples.append(data.Example.fromlist([line[1], line[0]], fields))

            with open(os.path.join(path, 'test.csv'), errors='ignore') as f:
                row_count_test = sum(1 for row in csv.reader(f, delimiter=','))

            with open(os.path.join(path, 'test.csv'), errors='ignore') as f:
                csv_reader = csv.reader(f, delimiter=',')

                test_examples = []
                for line in tqdm(csv_reader, total=row_count_test,desc="Loading test data", leave=True, position=0):
                    test_examples.append(data.Example.fromlist([line[1], line[0]], fields))

            examples={"train": train_examples, "test": test_examples}
        super(YELP, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, preprocess_txt=False, default_split=True, dev_ratio=.1, shuffle=True, root='.data', **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root, 'yelp')
        examples = cls(text_field, label_field, preprocess_txt=preprocess_txt, path=path, **kwargs).examples

        if default_split:
            if shuffle:
                random.shuffle(examples['train'])
                random.shuffle(examples['test'])

            return (cls(text_field, label_field, preprocess_txt=preprocess_txt, examples=examples['train']),
                    cls(text_field, label_field, preprocess_txt=preprocess_txt, examples=examples['test']))
        else:
            examples = examples['train'] + examples['test']
            if shuffle: random.shuffle(examples)
            dev_index = -1 * int(dev_ratio * len(examples))

            return (cls(text_field, label_field, preprocess_txt=preprocess_txt, examples=examples[:dev_index]),
                    cls(text_field, label_field, preprocess_txt=preprocess_txt, examples=examples[dev_index:]))

class SST2GLUE(TarDataset):
    url = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8'
    filename = 'SST-2.zip'
    dirname = 'SST-2'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, preprocess_txt, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        def clean_text_sst(string):
            """
            Tokenization/string cleaning for the SST dataset
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)  # replace anything not included with space
            string = re.sub(r"\s{2,}", " ", string)  # replace two or more {2,} spaces \s with a sinfle space
            return string.lower().strip().split(' ')  # string.strip().lower()

        if preprocess_txt:
            text_field.preprocessing = clean_text_sst
        #else:
        #	text_field.preprocessing = lambda s: s.split()

        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            # examples = {}
            # train_examples = []
            # test_examples = []
            # dev_examples = []
            # Load Train data
            with open(os.path.join(path, 'train.tsv'), errors='ignore') as f:
                row_count_train = sum(1 for row in csv.reader(f, delimiter='\t'))

            with open(os.path.join(path, 'train.tsv'), errors='ignore') as f:
                csv_reader = csv.reader(f, delimiter="\t")
                f.readline() # ignore headers
                train_examples=[]
                for line in tqdm(csv_reader, total=row_count_train, desc="Loading train data", leave=True, position=0):
                    train_examples.append(data.Example.fromlist([line[0], line[1]], fields))
            # Load dev data
            with open(os.path.join(path, 'dev.tsv'), errors='ignore') as f:
                row_count_test = sum(1 for row in csv.reader(f, delimiter='\t'))

            with open(os.path.join(path, 'dev.tsv'), errors='ignore') as f:
                csv_reader = csv.reader(f, delimiter='\t')
                f.readline()  # ignore headers
                dev_examples = []
                for line in tqdm(csv_reader, total=row_count_test,desc="Loading dev data", leave=True, position=0):
                    dev_examples.append(data.Example.fromlist([line[0], line[1]], fields))

            examples={"train": train_examples, "dev": dev_examples}
        super(SST2GLUE, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, preprocess_txt=False, default_split=True, dev_ratio=.1, shuffle=True, root='.data', **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, preprocess_txt=preprocess_txt, path=path, **kwargs).examples

        if default_split:
            if shuffle:
                random.shuffle(examples['train'])
                random.shuffle(examples['dev'])

            return (cls(text_field, label_field, preprocess_txt=preprocess_txt, examples=examples['train']),
                    cls(text_field, label_field, preprocess_txt=preprocess_txt, examples=examples['dev']))
        else:
            print('Not Implemented!! SST2_glue')
            # examples = examples['train'] + examples['test']
            # if shuffle: random.shuffle(examples)
            # dev_index = -1 * int(dev_ratio * len(examples))
            #
            # return (cls(text_field, label_field, preprocess_txt=preprocess_txt, examples=examples[:dev_index]),
            #
            #         cls(text_field, label_field, preprocess_txt=preprocess_txt, examples=examples[dev_index:]))

class SUBJ(TarDataset):
    url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz'
    filename = 'rotten_imdb.tar'
    dirname = 'rotten_imdb'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        # def clean_str(string):
        # 	"""
        # 	Tokenization/string cleaning for all datasets except for SST.
        # 	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        # 	"""
        # 	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        # 	string = re.sub(r"\'s", " \'s", string)
        # 	string = re.sub(r"\'ve", " \'ve", string)
        # 	string = re.sub(r"n\'t", " n\'t", string)
        # 	string = re.sub(r"\'re", " \'re", string)
        # 	string = re.sub(r"\'d", " \'d", string)
        # 	string = re.sub(r"\'ll", " \'ll", string)
        # 	string = re.sub(r",", " , ", string)
        # 	string = re.sub(r"!", " ! ", string)
        # 	string = re.sub(r"\(", " \( ", string)
        # 	string = re.sub(r"\)", " \) ", string)
        # 	string = re.sub(r"\?", " \? ", string)
        # 	string = re.sub(r"\s{2,}", " ", string)
        # 	return string.strip()
        #
        # text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join('quote.tok.gt9.5000'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'subjective'], fields) for line in f]
            with open(os.path.join('plot.tok.gt9.5000'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'objective'], fields) for line in f]
        super(SUBJ, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        """Create dataset objects for splits of the SUBJ dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio * len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))
