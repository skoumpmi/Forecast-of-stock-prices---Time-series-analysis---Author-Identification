import matplotlib.pyplot as plt
import argparse, os, glob
import numpy as np
import pandas as pd

def main():

    temp_pd = pd.DataFrame(columns=['Train_auc', 'Test_auc', 'Valid_auc'])

    acc_files=[]
    loss_files=[]
    auc_files=[]
    os.chdir(os.path.join(FLAGS.main_path, FLAGS.output_dir, TASK_NAME, 'csv'))
    for file in glob.glob('*.csv'):
        if 'acc' in file:
            acc_files.append(file)
        elif 'auc' in file:
            auc_files.append(file)
        else:
            loss_files.append(file)

    data_acc = {}
    fig = plt.figure()
    x_ticks_step=2
    df_test,df_val=temp_pd, temp_pd
    for file in acc_files:
        if 'test' in file:
            df_test = pd.read_csv(file, delimiter=',')['Value'][:FLAGS.n_epochs]
            df_test.index = df_test.index + 1
            data_acc.update({'Test_acc': df_test})
            ax=df_test.plot(label='test', style=['o-'])
            ax.set_xticks(np.arange(1, len(df_test.index)+1, step=x_ticks_step))
        elif 'val' in file:
            df_val = pd.read_csv(file, delimiter=',')['Value'][:FLAGS.n_epochs]
            df_val.index = df_val.index + 1
            data_acc.update({'Valid_acc': df_val})
            ax=df_val.plot(label='valid', style=['o-'])
            ax.set_xticks(np.arange(1, len(df_val.index)+1, step=x_ticks_step))
        else:
            df_train = pd.read_csv(file, delimiter=',')['Value'][:FLAGS.n_epochs]
            df_train.index = df_train.index + 1
            data_acc.update({'Train_acc': df_train})
            ax=df_train.plot(label='train', style=['o-'])
            ax.set_xticks(np.arange(1, len(df_train.index)+1, step=x_ticks_step))
    data_acc.update({'Epoch': range(len(df_train)) })

    # Create DataFrame
    data_acc_df = pd.DataFrame(data_acc)
    # creating a rank column and passing the returned rank series
    if not df_test.empty:
        data_acc_df["Test ACC Rank"] = data_acc_df["Test_acc"].rank(method ='min')
    if not df_val.empty:
        data_acc_df["Valid ACC Rank"] = data_acc_df["Valid_acc"].rank(method ='min')

    print(data_acc_df)

    plt.ylabel('Accuracy')
    plt.xlabel('epochs')
    plt.grid()
    plt.legend(loc='best')
    plt.legend()

    data_auc = {}
    fig = plt.figure()
    df_test,df_val=temp_pd,temp_pd
    for file in auc_files:
        if 'test' in file:
            df_test = pd.read_csv(file, delimiter=',')['Value'][:FLAGS.n_epochs]
            df_test.index = df_test.index + 1
            data_auc.update({'Test_auc': df_test})
            ax=df_test.plot(label='test', style=['o-'])
            ax.set_xticks(np.arange(1, len(df_test.index)+1, step=x_ticks_step))
        elif 'val' in file:
            df_val = pd.read_csv(file, delimiter=',')['Value'][:FLAGS.n_epochs]
            df_val.index = df_val.index + 1
            data_auc.update({'Valid_auc': df_val})
            ax=df_val.plot(label='valid', style=['o-'])
            ax.set_xticks(np.arange(1, len(df_val.index)+1, step=x_ticks_step))
        else:
            df_train = pd.read_csv(file, delimiter=',')['Value'][:FLAGS.n_epochs]
            df_train.index = df_train.index + 1
            data_auc.update({'Train_auc': df_train})
            ax=df_train.plot(label='train', style=['o-'])
            ax.set_xticks(np.arange(1, len(df_train.index)+1, step=x_ticks_step))

    data_auc.update({'Epoch': range(len(df_train))})
    # Create DataFrame
    data_auc_df = pd.DataFrame(data_auc)
    if not df_test.empty:
        data_auc_df["Test AUC Rank"] = data_auc_df["Test_auc"].rank(method ='min')
    if not df_val.empty:
        data_auc_df["Valid AUC Rank"] = data_auc_df["Valid_auc"].rank(method ='min')
    print(data_auc_df)


    plt.ylabel('AUC')
    plt.xlabel('epochs')
    plt.grid()
    plt.legend(loc='best')
    plt.legend()

    fig = plt.figure()
    for file in loss_files:
        if 'test' in file:
            df_test = pd.read_csv(file, delimiter=',')['Value'][:FLAGS.n_epochs]
            df_test.index = df_test.index + 1
            ax=df_test.plot(label='test', style=['o-'])
            ax.set_xticks(np.arange(1, len(df_test.index)+1, step=x_ticks_step))
        elif 'val' in file:
            df_val = pd.read_csv(file, delimiter=',')['Value'][:FLAGS.n_epochs]
            df_val.index = df_val.index + 1
            ax=df_val.plot(label='valid', style=['o-'])
            ax.set_xticks(np.arange(1, len(df_val.index)+1, step=x_ticks_step))
        else:
            df_train = pd.read_csv(file, delimiter=',')['Value'][:FLAGS.n_epochs]
            df_train.index = df_train.index + 1
            ax=df_train.plot(label='train', style=['o-'])
            ax.set_xticks(np.arange(1, len(df_train.index)+1, step=x_ticks_step))

    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.grid()
    plt.legend(loc='best')
    plt.legend()

    plt.show(block=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### Required parameters
    parser.add_argument("--output_dir", type=str, default='./', required=True,
                        help="The directory where algorithm experimnts exist.")
    parser.add_argument("--n_epochs", type=int, default=10, required=True,
                        help="The number of epochs to plot.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task ('yelp' or 'sst-2' or 'sst-5' or 'imdb') to train. ")
    parser.add_argument("--main_path", type=str, default='./', required=False,
                        help="The main path of the directory where algorithm experimnts exist.")
    # example: python plot_tensorboard.py --output_dir bert_outputs\fine_tuned --task_name imdb --n_epochs 10 --main_path G:\

    FLAGS = parser.parse_args()

    TASK_NAME = FLAGS.task_name
    # OUTPUT_FOLDER = os.path.join(FLAGS.main_path, FLAGS.output_dir, TASK_NAME)

    main()