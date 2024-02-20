from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm, trange
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torchtext import datasets
from torchtext import vocab
from torchtext import data
from CNN_txt import CNN2d
from glob import glob
import argparse, os
from utils import *
import mydatasets
import random
import torch
import json

SEED = 1234

def main():

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_valid_loss = []
    epoch_valid_acc = []
    epoch_test_loss = []
    epoch_test_acc = []
    epoch_valid_roc_auc = []
    epoch_test_roc_auc = []
    epoch_train_roc_auc = []


    BATCH_SIZES = (40, 256, 256)
    train_dl, valid_dl, test_dl, criterion, TEXT, LABEL = load_cnn_data(task_name=TASK_NAME,
                                                                 cache_folder=CACHE_DIR,
                                                                 device=DEVICE,
                                                                 batch_sizes=BATCH_SIZES,
                                                                embeddings_file=EMBEDDINGS_FILE,
                                                                embeddings_path=os.path.join(CACHE_DIR,
                                                                                             'embeddings'))

    train_dl, valid_dl, test_dl, criterion, TEXT, LABEL = load_cnn_data(task_name=TASK_NAME,
                                                                cache_folder=CACHE_DIR,
                                                                device=DEVICE,
                                                                batch_sizes=BATCH_SIZES,
                                                                embeddings_file=EMBEDDINGS_FILE,
                                                                embeddings_path=os.path.join(CACHE_DIR, 'embeddings'))

    criterion = criterion.to(DEVICE)
    # epoch_dirs=glob(os.path.join(FLAGS.output_dir, TASK_NAME, '*'))
    # epoch_dirs=[dir_ for dir_ in epoch_dirs if 'epoch_' in dir_]
    # epoch_dirs=epoch_dirs[:FLAGS.n_epochs+1]

    #t_bar = trange(0, len(epoch_dirs), desc='Epoch Iteration', leave=True)
    for epoch in trange(0, FLAGS.n_epochs, desc='Epoch Iteration', leave=True):

        restore_point="epoch_{0}".format(epoch+1)


        model = load_model(device=DEVICE,
                           restore_point=restore_point,
                           output_folder=OUTPUT_FOLDER,
                           config_name=CONFIG_NAME,
                           model_name=MODEL_NAME,
                           for_training=False)

        model = model.to(DEVICE)
        #model, optimizer, parameters=load_model(os.path.join(FLAGS.output_dir, FLAGS.task_name, dir_))

        train_loss, train_acc, train_roc_auc = evaluate(model, train_dl, criterion)
        valid_loss, valid_acc, valid_roc_auc  = evaluate(model, valid_dl, criterion)
        test_loss, test_acc, test_roc_auc = evaluate(model, test_dl, criterion)

        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_train_roc_auc.append(train_roc_auc)
        epoch_valid_loss.append(valid_loss)
        epoch_valid_acc.append(valid_acc)
        epoch_test_loss.append(test_loss)
        epoch_test_acc.append(test_acc)
        epoch_valid_roc_auc.append(valid_roc_auc)
        epoch_test_roc_auc.append(test_roc_auc)

        print(f'Epoch: {epoch:02}')
        print(f'\t Train. Loss: {train_loss:.3f} |  Train. Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc * 100:.2f}%')
        print(f'\t Val. AUC: {valid_roc_auc:.3f} |  Test. AUC: {test_roc_auc:.3f}')

    fig = plt.figure()
    #fig.suptitle('Accuracy')
    plt.plot(epoch_train_acc, label='train')
    plt.plot(epoch_valid_acc, label='valid')
    plt.plot(epoch_test_acc, label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('epochs')
    plt.grid()
    plt.legend(loc='best')

    fig = plt.figure()
    #fig.suptitle('Loss')
    plt.plot(epoch_train_loss, label='valid')
    plt.plot(epoch_valid_loss, label='valid')
    plt.plot(epoch_test_loss, label='test')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.grid()
    plt.legend(loc='best')

    fig = plt.figure()
    #fig.suptitle('AUC')
    plt.plot(epoch_train_roc_auc, label='train')
    plt.plot(epoch_valid_roc_auc, label='valid')
    plt.plot(epoch_test_roc_auc, label='test')
    plt.ylabel('AUC')
    plt.xlabel('epochs')
    plt.grid()
    plt.legend(loc='best')

    plt.show(block=True)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0
    y_true = []
    y_score = []

    model.eval()
    with torch.no_grad():

        for X, y in tqdm(iterator, desc="Evaluate Iteration", leave=True, position=0):

            predictions = model(X)#.squeeze(1)
            loss = criterion(predictions, y.long())
            acc, max_preds, max_preds_index = categorical_accuracy(predictions, y.long(), device=DEVICE)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            y_true.extend(y.cpu().data.numpy())
            y_score.extend(max_preds_index.cpu().data.numpy())

    # Compute fpr, tpr, thresholds and roc auc
    #fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = multiclass_roc_score(y_true, y_score, average="macro")
    # roc_auc = roc_auc_score(y_true, y_score)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), roc_auc

if __name__ == "__main__":

    LABEL = data.LabelField(dtype=torch.long, sequential=False, unk_token=None)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZES = (40, 256, 256)

    parser = argparse.ArgumentParser()

    ### Required parameters
    parser.add_argument("--output_dir", type=str, default='./', required=True,
                        help="The main path where all files will be stored.")
    parser.add_argument("--cache_dir", type=str, default='cache', required=False,
                        help="The main path where all files will be stored.")
    parser.add_argument("--n_epochs", type=int, default=10, required=True,
                        help="The main path where all files will be stored.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task ('yelp' or 'sst-2' or 'sst-5' or 'imdb') to train. ")
    parser.add_argument("--gpu_mode", type=str2bool, default=True, required=True, help="To run in gpu or not.")
    parser.add_argument("--embeddings_file", type=str, default="GoogleNews-vectors-negative300_pytorch.txt",
                        required=False, help="The filename of the pre-computed embeddings.")
    parser.add_argument("--main_path", type=str, default='./', required=False,
                        help="The main path where all files will be stored.")

    FLAGS           =   parser.parse_args()

    EMBEDDINGS_FILE = FLAGS.embeddings_file
    TASK_NAME       =   FLAGS.task_name
    CACHE_DIR       =   os.path.join(FLAGS.main_path, "cache/")
    CONFIG_NAME     =   "config.json"
    MODEL_NAME      =   "pytorch_model.bin"
    OUTPUT_FOLDER = os.path.join(FLAGS.output_dir, TASK_NAME)

    print('--------------------------------------')
    print("CHECK POINT FOLDER {0}".format(os.path.join(FLAGS.output_dir, FLAGS.task_name)))

    print('--------------------------------------')

    main()

