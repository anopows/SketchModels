import numpy as np
import tensorflow as tf
import glob
import csv
import datetime 

from sketch_util import create_snapshot_imgs

# Folders for storage/retrival
main_directory  = '../'
data_directory  = main_directory + 'kaggle_data/'
train_folder    = data_directory + 'train/'
valid_folder    = data_directory + 'valid/'
test_folder     = data_directory + 'test/'
train_folder_small    = data_directory + 'train_small/'
valid_folder_small    = data_directory + 'valid_small/'
test_folder_small     = data_directory + 'test_small/'

def _populate_folders(main_directory):
    data_directory        = main_directory + 'kaggle_data/'
    train_folder          = data_directory + 'train/'
    valid_folder          = data_directory + 'valid/'
    test_folder           = data_directory + 'test/'
    train_folder_small    = data_directory + 'train_small/'
    valid_folder_small    = data_directory + 'valid_small/'
    test_folder_small     = data_directory + 'test_small/'
    
def class_names(small=False, escape_spaces=True, base_folder=None):
    if base_folder: _populate_folders(base_folder)
    class_name = 'classnames'
    if small: class_name += '_small'
    
    class_names = [] # 345 classes, small: 10 classes
    with open(data_directory + class_name + '.csv', 'r') as cln_file:
        for line in cln_file:
            cl_name = line[:-1]
            if escape_spaces:
                cl_name = cl_name.replace(' ', '_') # escape spaces for kaggle
            class_names.append(cl_name)
    return class_names

def get_ith_name(id, small=False, escape_spaces=True, base_folder=None):
    return class_names(small, escape_spaces, base_folder)[id]

def write_results(keys, solutions, model_name=None, path='../solutions/'):
    if not model_name:
        now = datetime.datetime.now()
        if path[-1] != '/': path += '/'
        file_path = path + now.strftime('_%Y-%m-%dT%H:%M:%S') + '.csv'
    with open(file_path, 'w', newline='') as csvfile:
        fnames = ['key_id', 'word']
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fnames)
        writer.writeheader()
        for i,sol in enumerate(solutions):
            names = [get_ith_name(s) for s in sol]
            names = ' '.join(names)
            writer.writerow({'key_id': str(keys[i]), 'word': names})

def _parse_tfexample_fn(example_proto, test=False, max_seqlen=None):
    # Define data format in tfrecord
    features={
            'key_id':  tf.FixedLenFeature([], dtype=tf.int64),
            'country': tf.FixedLenFeature([], dtype=tf.string),
            'sketch': tf.VarLenFeature(dtype=tf.int64),
    }
    if not test:
        features['label'] = tf.FixedLenFeature([], dtype=tf.string)

    tfrecord_features = tf.parse_single_example(
        example_proto, features=features, name='features')

    # Recreate data
    key_id  = tfrecord_features['key_id']
    country = tfrecord_features['country']
    sketch = tf.sparse_tensor_to_dense(tfrecord_features['sketch'])
    sketch = tf.reshape(sketch, shape=[-1,3])
    length = tf.shape(sketch)[0]
    if max_seqlen: # limit max length
        length = tf.minimum(length, max_seqlen)
        sketch = sketch[:length]

    return_vals = [key_id, country, length, sketch]
    if not test:
        label = tfrecord_features['label']
        return_vals.append(label)
    
    return return_vals

def _get_int_labels(labels, small):
    with tf.variable_scope('label_mapping', reuse=tf.AUTO_REUSE):
        if not small:
            class_names = [] # 340 classes
            with open(data_directory + 'classnames.csv', 'r') as cln_file:
                for line in cln_file:
                    class_names += [line[:-1]]
        else:
            class_names = [] # 5 fruits, 5 vehicles
            with open(data_directory + 'classnames_small.csv', 'r') as cln_file:
                for line in cln_file:
                    class_names += [line[:-1]]
        # label name -> id
        mapping_strings = tf.get_variable('class_names', shape=[len(class_names)], dtype=tf.string, 
                                          initializer=tf.constant_initializer(class_names),
                                          trainable=False
                          )
        table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings)
        ids  = table.lookup(labels)
        return ids

def train_data(batch_size=30, num_prefetch=2, 
               epochs=None, small=False,
               seed=42, base_folder=None,
               with_imgs=False,
               num_snapshots=6,
               max_seqlen=None):   
    """ Return batches of training data
    
    Returns:
        key id
        country code
        length of sketch
        sketch
        (image)
        label
    """
    if small: 
        file_pattern = train_folder_small + '*.tfrecords'
    else:
        file_pattern = train_folder + '*.tfrecords'
    return get_data(file_pattern, batch_size=batch_size, small=small, 
                    num_prefetch=num_prefetch, epochs=epochs, 
                    seed=seed, base_folder=base_folder,
                    with_imgs=with_imgs, num_snapshots=num_snapshots,
                    max_seqlen=max_seqlen)

def valid_data(batch_size=30, num_prefetch=2, 
              epochs=1, small=False,
              seed=42, base_folder=None,
              with_imgs=False,
              num_snapshots=6,
              max_seqlen=None):
    """ Return batches of validation data
    
    Returns:
        key id
        country code
        length of sketch
        sketch
        (image)
        label
    """
    if small: 
        file_pattern = valid_folder_small + '*.tfrecords'
    else:
        file_pattern = valid_folder + '*.tfrecords'
    return get_data(file_pattern, batch_size=batch_size, small=small, 
                    num_prefetch=num_prefetch, epochs=epochs,
                    seed=seed, base_folder=base_folder, with_imgs=with_imgs, num_snapshots=num_snapshots,
                    max_seqlen=max_seqlen)
                   
def test_data(batch_size=30, num_prefetch=2,
              epochs=1, small=False,
              seed=42, base_folder=None,
              with_imgs=False,
              num_snapshots=6,
              max_seqlen=None):
    """ Return batches of test data
    
    Returns:
        key id
        country code
        length of sketch
        sketch
        (image)
    """
    assert small == False
    file_pattern = test_folder + '*.tfrecords'
    return get_data(file_pattern, batch_size=batch_size, small=False,
                    num_prefetch=num_prefetch, epochs=epochs,
                    seed=seed, base_folder=base_folder,
                    test_data=True, with_imgs=with_imgs, num_snapshots=num_snapshots,
                    max_seqlen=max_seqlen)
    

def get_data(file_pattern=train_folder + '*.tfrecords', 
             batch_size=30, 
             num_prefetch=2, 
             small=False,
             epochs=None,
             seed=None,
             base_folder=None,
             test_data=False,
             with_imgs=False,
             num_snapshots=6,
             max_seqlen=None):

    """ Main function to generate samples from tfrecords
    
    Returns:
        key_id
        country
        length
        sketch
        (image)
        (label)
    """
    if base_folder: _populate_folders(base_folder)

    with tf.variable_scope('input_pipepline'):
        dataset = tf.data.Dataset.list_files(file_pattern=file_pattern, shuffle=False) # TF >=1.9: shuffle: True but with seed
        num_datasets = len(glob.glob(file_pattern))
        assert num_datasets > 0, file_pattern

        block_length = 1
        # Have num_datasets tfrecords/classes and take 2 entries from each class at a time.
        # Also shuffle datasets before so no same 2 values are taken in later epoch
        dataset = dataset.interleave( 
            lambda x: tf.data.TFRecordDataset(x).repeat(epochs).shuffle(5000, seed=seed), # 
            cycle_length=num_datasets, # Interleave from all classes
            block_length=block_length)   # same classes at a time
        dataset = dataset.map( # 
            lambda x: _parse_tfexample_fn(x, test=test_data, max_seqlen=max_seqlen),
            num_parallel_calls=2 
        )
        dataset = dataset.shuffle(5000, seed=seed)

        # create imgs from sketches 
        if with_imgs:
            def _img_transform(key_id, country, length, sketch, *labels):
                img_fn  = lambda length, sketch: create_snapshot_imgs(length, sketch, num_snapshots=num_snapshots)
                output = tf.py_func(img_fn, [length, sketch], [tf.int32, tf.uint8, tf.int32], stateful=False)
                num_imgs    = output[0]
                imgs        = tf.reshape(output[1], (num_snapshots,224*224))
                img_indices = tf.reshape(output[2], (-1,))
                return (key_id, country, length, sketch, num_imgs, imgs, img_indices, *labels)
                
            dataset = dataset.map(_img_transform, num_parallel_calls=4)

        data_shape = ([], [], [], [max_seqlen,3]) # key_id, country, length, sketch: [maxlen x 3]
        if with_imgs:     data_shape += ([], [num_snapshots, 224*224], [max_seqlen])
        if not test_data: data_shape += ([],)

        dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=data_shape)
        dataset = dataset.prefetch(num_prefetch)

        if not test_data:
            *args, labels = dataset.make_one_shot_iterator().get_next()
            labels = _get_int_labels(labels, small=small)
            return args + [labels] 
        else: 
            return dataset.make_one_shot_iterator().get_next()
