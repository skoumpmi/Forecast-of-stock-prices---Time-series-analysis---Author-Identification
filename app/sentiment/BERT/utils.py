# from pytorch_transformers import BertTokenizer, BertForPreTraining
from transformers import *
from sklearn.metrics import roc_auc_score, accuracy_score
from bert_classifiers import linearClassifier
#from pytorch_transformers import BertConfig
from transformers import *
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn import preprocessing
from torchtext import datasets
from torchtext import vocab
from torchtext import data
from CNN_txt import CNN2d
import torch.nn as nn
import pandas as pd
import mydatasets
import argparse
import string
import random
import torch
import json
import nltk
import re
import os

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

SEED = 1234

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

def simple_tokenize(s):
    return s.split(' ')

def strip_tokenize(s):
    return s.strip()

def clean_str_imdb(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower().split(' ')

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) # replace anything not included with space
    string = re.sub(r"\s{2,}", " ", string) # replace two or more {2,} spaces \s with a sinfle space
    return string.lower().strip().split(' ')#string.strip().lower()

def clean_str_except_sst(string):
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
    return string.lower().strip().split(' ') #string.strip()

class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x and y

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper
            y = getattr(batch, self.y_vars)
            yield (x, y)

    def __len__(self):
        return len(self.dl)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def multiclass_roc_score(y_true, y_score, average="macro"):

    lb = preprocessing.LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_score = lb.transform(y_score)

    try: #Only one class present in y_true. ROC AUC score is not defined in that case.
        return roc_auc_score(y_true, y_score, average=average)
    except ValueError as e:
        print ( '=====>>>>>>> ' + str(e) + '<<<<<<=====')
        return accuracy_score(y_true, y_score)

def load_cnn_model(device, restore_point, output_folder, config_name, model_name, gpu_enabled=False):

    print("=> Loading model {0}...".format(restore_point))

    checkpoint_model = torch.load(os.path.join(output_folder, restore_point, 'pytorch_model.bin'))

    with open(os.path.join(output_folder, restore_point, 'config.json')) as f:
        checkpoint_config = json.load(f)

    print ('Modek parameters {0}'.format(checkpoint_config))


    model = CNN2d(n_filters=checkpoint_config["n_filters"],
                filter_sizes=checkpoint_config["filter_sizes"],
                multi_channel=checkpoint_config["multi_channel"],
                output_dim=checkpoint_config["output_dim"],
                dropout=checkpoint_config["dropout"],
                pad_idx=checkpoint_config["pad_idx"],
                vocab_size=checkpoint_config["vocab_size"],
                embedding_dim=checkpoint_config["embedding_dim"])


    model.load_state_dict(checkpoint_model['model_state_dict'])

    model.cuda() if gpu_enabled else model.cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda() if gpu_enabled else v.cpu()
    #
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    # optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])

    start_epoch = checkpoint_model['epoch'] + 1
    parameters = filter(lambda p: p.requires_grad, model.parameters())


    return model, optimizer, parameters, start_epoch

def save_cnn_checkpoint(model, optimizer, epoch, output_folder, config_name, model_name, save_model):

    if not save_model:
        return

    """Save checkpoint if a new best is achieved"""
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}

    file_path = os.path.join(output_folder, 'epoch_{0}'.format(epoch))
    print("=> Saving model stats in {0}".format(file_path))
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    model.config_json_file(os.path.join(file_path, config_name))

    torch.save(state, os.path.join(file_path, model_name))

def load_bert_check_point(model, optimizer, scheduler, restore_point, output_folder):

    print("=> Loading model {0}...".format(restore_point))

    checkpoint_model = os.path.join(output_folder, restore_point, 'pytorch_model.bin')
    checkpoint_config = os.path.join(output_folder, restore_point, 'config.json')

    checkpoint_model = torch.load(checkpoint_model)

    model.load_state_dict(checkpoint_model["model_state_dict"])

    optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])

    scheduler.load_state_dict(checkpoint_model["scheduler_state_dict"])

    return model, optimizer, scheduler

def load_bert_check_point_model(restore_point, output_folder, model_type=None):

    print("=> Loading model {0}...".format(restore_point))

    checkpoint_model_name = os.path.join(output_folder, restore_point, 'pytorch_model.bin')
    checkpoint_config_name = os.path.join(output_folder, restore_point, 'config.json')

    checkpoint_model = torch.load(checkpoint_model_name)

    if model_type:
        #Load pre-trained model (weights)
        config = BertConfig().from_json_file(checkpoint_config_name)
        model = model_type(config)
        model.load_state_dict(checkpoint_model['model_state_dict'])
        return model, checkpoint_model['epoch']
    else:
        return checkpoint_model['model_state_dict'], checkpoint_model['epoch']

def load_bert_check_point_optimizer(optimizer, restore_point, output_folder):

    print("=> Loading optimizer {0}...".format(restore_point))

    checkpoint_model_name = os.path.join(output_folder, restore_point, 'pytorch_model.bin')
    checkpoint_model = torch.load(checkpoint_model_name)
    optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])

    return optimizer

def load_bert_check_point_scheduler(scheduler, restore_point, output_folder):

    print("=> Loading scheduler {0}...".format(restore_point))
    checkpoint_model_name = os.path.join(output_folder, restore_point, 'pytorch_model.bin')
    checkpoint_model = torch.load(checkpoint_model_name)
    scheduler.load_state_dict(checkpoint_model["scheduler_state_dict"])

    return scheduler

def save_bert_checkpoint(model, optimizer, scheduler, epoch, output_folder):

    file_path = os.path.join(output_folder, 'epoch_{0}'.format(epoch))
    print("=> Saving model stats in {0}".format(file_path))
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    output_model_file = os.path.join(file_path, 'pytorch_model.bin')
    output_config_file = os.path.join(file_path, 'config.json')

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    """Save checkpoint if a new best is achieved"""
    state = {'epoch': epoch+1,
             'model_state_dict': model_to_save.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict()}

    torch.save(state, output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    ###################################################################################################################

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = './model_save/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    ###################################################################################################################



def data_to_csv(data, file_name, label_field):

    if not data: return

    import csv
    with open(file_name, 'w+', encoding="utf-8") as outcsv:
        # configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        #writer.writerow(['number', 'text', 'number'])
        for item in data.examples:
            writer.writerow([label_field.vocab.stoi[item.label], ' '.join(item.text)])

def convert_bin_emb_txt(cache, emb_file):

    import gensim
    from gensim.models import KeyedVectors
    # txt_name = os.path.basename(emb_file).split(".")[0] +".txt"
    # emb_txt_file = os.path.join(cache, txt_name)
    emb_bin_file = os.path.join(cache, emb_file)
    emb_model = KeyedVectors.load_word2vec_format(emb_bin_file, binary=True, encoding="ISO-8859-1", unicode_errors='ignore')
    word2index = {token: token_index for token_index, token in enumerate(emb_model.index2word)}
    # emb_model.save_word2vec_format(emb_txt_file, binary=True, write_first_line=False)
    return emb_model.vectors, word2index, emb_model.vector_size

def TEXT_obj(task_name, save_data=True):

    lower = False if save_data else True

    if task_name == 'yelp':
        tokenize = None if save_data else lambda s: s.strip()
        TEXT = data.Field(tokenize=tokenize, lower=lower)
    elif task_name == 'imdb':
        tokenize = None if save_data else clean_str_imdb
        TEXT = data.Field(tokenize=tokenize, lower=lower)
    elif task_name == 'MR':
        tokenize = None if save_data else lambda s: s.strip()
        TEXT = data.Field(tokenize=tokenize, lower=lower)
    elif task_name in ['sst-2-standard', 'sst-5-standard']:
        tokenize = None if save_data else clean_str_sst
        pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        TEXT = data.Field(use_vocab=False, tokenize=tokenizer.encode, pad_token=pad_index,lower=lower)
    # elif task_name == 'sst-5':
    #     tokenize = None if save_data else clean_str_sst
    #     TEXT = data.Field(tokenize=tokenize, lower=lower)
    elif task_name == 'sst-2-glue':
        tokenize = None if save_data else lambda s: s.strip()
        TEXT = data.Field(tokenize=lambda s: s.strip(), lower=lower)
    else:
        raise ValueError('This task is not included!!!')

    return TEXT

def pytorch_data(text_field, label_field, task_name, cache_folder, valid_data=False, save_data=False):

    # LABEL = data.LabelField(dtype=torch.long, sequential=False, unk_token=None)
    # TEXT=TEXT_obj(task_name, save_data=save_data)
    preprocess_txt = False if save_data else True
    train_data, test_data, valid_data=None, None, None

    if task_name == 'yelp':
        print('DATASET YELP loaded in {0}'.format(os.path.join(cache_folder, '.data')))
        train_data, test_data = mydatasets.YELP.splits(text_field, label_field,
                                                       preprocess_txt=preprocess_txt,
                                                       default_split=True,
                                                       shuffle=False,
                                                       root=os.path.join(cache_folder, '.data'))
        if valid_data:
            train_data, valid_data = train_data.split(split_ratio=0.9, random_state=random.seed(SEED))

    elif task_name == 'imdb':
        print('DATASET IMDB loaded in {0}'.format(os.path.join(cache_folder, '.data')))
        train_data, test_data = datasets.IMDB.splits(text_field, label_field, root=os.path.join(cache_folder, '.data'))

        if valid_data:
            train_data, valid_data = train_data.split(split_ratio=0.9, random_state=random.seed(SEED))

    elif task_name == 'MR':
        print('DATASET MR loaded in {0}'.format(os.path.join(cache_folder, '.data')))
        train_data, test_data = mydatasets.MR.splits(text_field, label_field,
                                                     preprocess_txt=preprocess_txt,
                                                     root=os.path.join(cache_folder, '.data', 'MR'))
        if valid_data:
            train_data, valid_data = train_data.split(split_ratio=0.9, random_state=random.seed(SEED))

    elif task_name == 'sst-2-standard':
        print('DATASET SST-2 loaded in {0}'.format(os.path.join(cache_folder, '.data', 'sst-2-standard')))
        train_data, valid_data, test_data = datasets.SST.splits(text_field, label_field,
                                                                fine_grained=False,
                                                                train_subtrees=False,
                                                                filter_pred=lambda ex: ex.label != 'neutral',
                                                                root=os.path.join(cache_folder, '.data', 'sst-2-standard'))
    elif task_name == 'sst-5-standard':
        print('DATASET SST-5 loaded in {0}'.format(os.path.join(cache_folder, '.data', 'sst-5-standard')))
        train_data, valid_data, test_data = datasets.SST.splits(text_field, label_field,
                                                                fine_grained=True,
                                                                # filter_pred=lambda ex: ex.label != 'neutral',
                                                                train_subtrees=False,
                                                                root=os.path.join(cache_folder, '.data', 'sst-5-standard'))

    elif task_name == 'sst2-glue':
        print('DATASET SST-2 GLUE loaded in {0}'.format(os.path.join(cache_folder, '.data')))
        train_data, valid_data = mydatasets.SST2GLUE.splits(text_field, label_field,
                                                     preprocess_txt=preprocess_txt,
                                                     root=os.path.join(cache_folder, '.data'))

        # if valid_data:
        #     train_data, valid_data = train_data.split(split_ratio=0.9, random_state=random.seed(SEED))
        # else:
        #     valid_data=None
    else:
        raise ValueError('This task is not included!!!')

    return train_data, test_data, valid_data

def save_data(task_name, cache_folder):

    LABEL = data.LabelField(dtype=torch.long, sequential=False, unk_token=None)
    TEXT = TEXT_obj(task_name, save_data=True)
    train_data, test_data, valid_data = pytorch_data(TEXT, LABEL, task_name, cache_folder, save_data=True)

    if task_name == 'sst2-glue': return

    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    data_to_csv(train_data, os.path.join(cache_folder, '.data', task_name, 'train.csv'), LABEL)
    data_to_csv(test_data,  os.path.join(cache_folder, '.data', task_name, 'test.csv'), LABEL)
    data_to_csv(valid_data, os.path.join(cache_folder, '.data', task_name, 'dev.csv'), LABEL)

    vocab = TEXT.vocab
    path = os.path.join(cache_folder, '.data', task_name)

    with open(path, 'w+') as f:
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}')

def load_cnn_data(task_name, cache_folder, device, batch_sizes, embeddings_file, embeddings_path):


    TEXT = TEXT_obj(task_name)

    print('==> Loading data...')
    LABEL = data.LabelField(dtype=torch.long, sequential=False, unk_token=None)
    TEXT = TEXT_obj(task_name, save_data=False)
    train_data, test_data, valid_data = pytorch_data(TEXT, LABEL, task_name, cache_folder, save_data=False)

    n_train_samples = len(train_data.examples) if train_data else 0
    n_test_samples = len(test_data.examples) if test_data else 0
    n_valid_samples = len(valid_data.examples) if valid_data else 0

    print('==> Training size {0}, Validation size {1}, Test size {2}'.format(n_train_samples,
                                                                         n_valid_samples,
                                                                         n_test_samples))
    print('==> Build Vocabulary...')
    vectors, stoi, dim  = convert_bin_emb_txt(embeddings_path, embeddings_file)
    #MAX_VOCAB_SIZE = 161850
    TEXT.build_vocab(train_data)
    TEXT.vocab.set_vectors(stoi, torch.from_numpy(vectors).float().to(device), dim)
    LABEL.build_vocab(train_data)

    print('==> Pretrain vocab length {0}'.format(len(TEXT.vocab)))

    print('==> Execution on {0}'.format(device))

    train_iterator, valid_iterator, test_iterator = None, None, None
    train_dl, test_dl, valid_dl = None, None, None
    if valid_data and test_data:
        datasets=(train_data, valid_data, test_data)
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            datasets=datasets,
            batch_sizes=batch_sizes,
            shuffle=True,
            device=device)
        valid_dl = BatchWrapper(valid_iterator, "text", "label")
        test_dl = BatchWrapper(test_iterator, "text", "label")
    elif valid_data and not test_data:
        datasets = (train_data, valid_data)
        train_iterator, valid_iterator = data.BucketIterator.splits(
            datasets=datasets,
            batch_sizes=batch_sizes,
            shuffle=True,
            device=device)
        valid_dl = BatchWrapper(valid_iterator, "text", "label")
    else:
        datasets = (train_data, test_data)
        train_iterator, test_iterator = data.BucketIterator.splits(
            datasets=datasets,
            batch_sizes=batch_sizes,
            shuffle=True,
            device=device)
        test_dl = BatchWrapper(test_iterator, "text", "label")

    train_dl = BatchWrapper(train_iterator, "text", "label")

    # 2 because of crossentropy https://stackoverflow.com/questions/53628622/loss-function-its-inputs-for-binary-classification-pytorch
    criterion = nn.CrossEntropyLoss()

    return train_dl, valid_dl, test_dl, criterion, TEXT, LABEL

def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def categorical_accuracy(preds, y, device):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds, max_preds_index = preds.max(dim=1)
    correct = max_preds_index.eq(y)
    if device == torch.device("cuda"):
        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = correct.sum().to(dtype=torch.float) / torch.cuda.LongTensor([y.shape[0]]).to(dtype=torch.float)
    else:
        result = correct.sum().to(dtype=torch.float) / torch.LongTensor([y.shape[0]]).to(dtype=torch.float)

    return result, max_preds, max_preds_index