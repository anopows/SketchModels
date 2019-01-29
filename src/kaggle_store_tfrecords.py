import numpy as np
import tensorflow as tf
import os
import csv
import ast
import joblib

from parse_binary import samples_from_binary
from parse_binary import _parse_sample
from sketch_io import class_names
from utility import in_jupyter

# Folders for storage/retrival
data_directory = '../kaggle_data/'
csv_folder      = data_directory + 'csvs/'
records_folder  = data_directory + 'class_records/'
train_folder    = data_directory + 'train/'
valid_folder    = data_directory + 'valid/'
test_folder     = data_directory + 'test/'

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def store_test_to_tfrecord():
    print("Storing test samples to tfrecord")
    if not os.path.exists(test_folder): os.makedirs(test_folder)

    with open('../kaggle_data/csvs/test_simplified.csv') as csvfile, \
         tf.python_io.TFRecordWriter(test_folder + 'test.tfrecords') as writer:

        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for i,cols in enumerate(reader):
            if (i+1)%5000 == 0: print("Read {} samples from test set".format(i+1))
            key_id       = int(cols[0])
            country_code = cols[1]
            sketch       = ast.literal_eval(cols[2])
            sketch       = _parse_sample(sketch).reshape([-1]) # to format [maxlen x 3]

            key_id_parsed        = _int64_feature([key_id]) 
            country_code_parsed  = _bytes_feature([bytes(country_code, 'utf-8')])
            sketch_parsed        = _int64_feature(sketch) # already np array, don't put in array

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'key_id':  key_id_parsed,
                        'country': country_code_parsed,
                        'sketch':  sketch_parsed
                    }
                )
            )
            writer.write(example.SerializeToString())

def store_class_to_tfrecord(classname):
    classname = classname.replace('_', ' ')
    print("Storing train sample {} to tfrecord".format(classname))
    if not os.path.exists(valid_folder): os.makedirs(valid_folder)
    if not os.path.exists(train_folder): os.makedirs(train_folder)

    with open('../kaggle_data/csvs/train/{}.csv'.format(classname)) as csv_file, \
         tf.python_io.TFRecordWriter(valid_folder + classname + '.tfrecords') as valid_writer, \
         tf.python_io.TFRecordWriter(train_folder + classname + '.tfrecords') as train_writer:

        reader = csv.reader(csv_file, delimiter=',')
        header = next(reader) # countrycode,drawing,key_id,recognized,timestamp,word
        for i,cols in enumerate(reader):
            # read from csv
            key_id       = int(cols[2])
            country_code = cols[0]
            sketch       = ast.literal_eval(cols[1])
            sketch       = _parse_sample(sketch).reshape([-1]) # to format [maxlen x 3]
            label        = cols[5]
            # parse for proto format
            key_id_parsed        = _int64_feature([key_id]) 
            country_code_parsed  = _bytes_feature([bytes(country_code, 'utf-8')])
            sketch_parsed        = _int64_feature(sketch) # already np array, don't put in array
            label_parsed         = _bytes_feature([bytes(label, 'utf-8')])

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'key_id':  key_id_parsed,
                        'country': country_code_parsed,
                        'sketch':  sketch_parsed,
                        'label':   label_parsed
                    }
                )
            )
            if i<5000: # First 5000 samples go to validaiton set
                valid_writer.write(example.SerializeToString())
            else:
                train_writer.write(example.SerializeToString())

def store_train_to_tfrecord():
    # In parallel write the tfrecords
    from joblib import Parallel, delayed
    Parallel(n_jobs=5)(map(delayed(store_class_to_tfrecord), class_names()))
    
    # # In sequence
    # for cln in class_names():
    #     store_class_to_tfrecord(cln)

def main():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    if in_jupyter(): 
        tf.app.flags.DEFINE_string('f', '', 'kernel')
    flags.DEFINE_string('store', 'train', "Which dataset to parse")    
    
    assert FLAGS.store in ['test', 'train']
    if FLAGS.store == 'test':
        store_test_to_tfrecord()
    elif FLAGS.store == 'train':
        store_train_to_tfrecord()

if __name__ == '__main__':
    main()