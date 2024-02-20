from pytorch_transformers import BertTokenizer
from multiprocessing import Pool, cpu_count
import convert_examples_to_features
from contextlib import closing
from tqdm import tqdm, trange
from tools import *
from utils import *
import argparse
import pickle

def prepare_tsv_files():
    try:
        print('==> Prepare data in {0}'.format(DATA_DIR))
        if FLAGS.task_name == 'sst2-glue':
            prepare_sst2_sst5_imdb_mr_data('train', path=DATA_DIR)
            prepare_sst2_sst5_imdb_mr_data('dev', path=DATA_DIR)
        elif FLAGS.task_name == 'yelp':
            prepare_yelp_data('train', path=DATA_DIR)
            prepare_yelp_data('test', path=DATA_DIR)
        elif FLAGS.task_name in ['imdb', 'MR', 'sst2-glue', 'sst-2-standard', 'sst-5-standard'] :
            prepare_sst2_sst5_imdb_mr_data('train', path=DATA_DIR)
            prepare_sst2_sst5_imdb_mr_data('test', path=DATA_DIR)
            if FLAGS.task_name in ['sst-2-standard', 'sst-5-standard']:
                prepare_sst2_sst5_imdb_mr_data('dev', path=DATA_DIR)

    except OSError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

def text_processor():

    if FLAGS.task_name == 'sst-2-glue':
        processor = SST2GLUEProcessor()
    elif FLAGS.task_name in ['yelp', 'imdb', 'MR', 'sst2-glue', 'sst-2-standard', 'sst-5-standard']:
        # processor = BinaryClassificationProcessor()
        processor = MulticlassClassificationProcessor()
    return processor

def load_tsv_files():

    processor = text_processor()

    train_examples = processor.get_train_examples(DATA_DIR)
    label_list = processor.get_labels()
    train_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    #########################################################################################
    # This line producess vocab.txt file
    train_tokenizer.save_vocabulary(OUTPUT_DIR)

    train_tokenizer.save_pretrained("C:/Users/annak/Desktop/cleanBERT")
    #########################################################################################


    try:
        processor = text_processor()
        eval_examples = processor.get_dev_examples(DATA_DIR)
    except IOError as e:
        eval_examples=None
        print("Loading dev_examples --> I/O error({0}): {1}".format(e.errno, e.strerror))
    try:
        processor = text_processor()
        test_examples = processor.get_test_examples(DATA_DIR)
    except IOError as e:
        test_examples=None
        print("Loading test_examples --> I/O error({0}): {1}".format(e.errno, e.strerror))

    external_tokenizer = BertTokenizer.from_pretrained(os.path.join(OUTPUT_DIR, 'vocab.txt'), do_lower_case=False)

    return train_examples, eval_examples, test_examples, label_list, train_tokenizer, external_tokenizer

def extract_features(examples, label_list, file_name, tokenizer):
    # tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)

    process_count = cpu_count() - 1
    examples_len = len(examples)
    print('Preparing to convert {0} examples..'.format(examples_len))
    print('Spawning {0} processes..'.format(process_count))

    label_map = {label: i for i, label in enumerate(label_list)}
    examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in
                               examples]

    with closing(Pool(processes=process_count)) as p:
        features = list(tqdm(p.imap(convert_examples_to_features.convert_example_to_feature,
                                    examples_for_processing), total=examples_len))

    with open(DATA_DIR + file_name + ".pkl", "wb") as f:
        pickle.dump(features, f)

def main():

    # 1. Prepare tsv files, BERT model work in this format
    prepare_tsv_files()
    # 2. Load tsv files
    train_examples, eval_examples, test_examples, label_list, train_tokenizer, external_tokenizer = load_tsv_files()
    # 3. Extract features, BERT model work in this format
    extract_features(train_examples, label_list, 'train_features_' + FLAGS.task_name, train_tokenizer)
    if eval_examples:
        extract_features(eval_examples, label_list, 'eval_features_' + FLAGS.task_name, external_tokenizer)
    if test_examples:
        extract_features(test_examples, label_list, 'test_features_' + FLAGS.task_name, external_tokenizer)

if __name__ == "__main__":

    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
    MAX_SEQ_LENGTH = 128
    OUTPUT_MODE = 'classification'

    parser = argparse.ArgumentParser()

    ### Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train")
    parser.add_argument("--main_path", type=str, default='./', required=False,
                        help="The main path where all files will be stored.")

    FLAGS = parser.parse_args()
    main_DATA_DIR = os.path.join(FLAGS.main_path, "cache",".data")
    # The input data dir. Should contain the .tsv files (or other data files) for the task.
    if FLAGS.task_name == 'sst-2-glue':
        DATA_DIR = os.path.join(main_DATA_DIR, "sst-2-glue/")
        TASK_NAME = 'sst-2-glue'
    elif FLAGS.task_name == 'sst-2-standard':
        DATA_DIR = os.path.join(main_DATA_DIR, "sst-2-standard/")
        TASK_NAME = 'sst-2-standard'
    elif FLAGS.task_name == 'sst-5-standard':
        DATA_DIR = os.path.join(main_DATA_DIR, "sst-5-standard//")
        TASK_NAME = 'sst-5-standard'
    elif FLAGS.task_name == 'yelp':
        DATA_DIR = os.path.join(main_DATA_DIR, "yelp/")
        TASK_NAME = 'yelp'
    elif FLAGS.task_name == 'imdb':
        DATA_DIR = os.path.join(main_DATA_DIR, "imdb/")
        TASK_NAME = 'imdb'
    elif FLAGS.task_name == 'MR':
        DATA_DIR = os.path.join(main_DATA_DIR, "MR/")
        TASK_NAME = 'MR'
    else:
        raise ValueError("Task name {0} does not exist".format(FLAGS.task_name))

    # The output directory where the fine-tuned model and checkpoints will be written.
    OUTPUT_DIR = os.path.join(main_DATA_DIR, TASK_NAME)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # save_data('sst-5-standard', OUTPUT_DIR)

    main()