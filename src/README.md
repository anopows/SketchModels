# Instructions

The code is developed using python 3.6 and Tensorflow Version 1.8

## Download and parse dataset into tfrecords

### Download dataset
Install kaggle-api and get your kaggle token
```bash
mkdir kaggle_data/csvs
cd <project_dir>/kaggle_data/csvs
kaggle competitions download -c quickdraw-doodle-recognition -f test_simplified.csv
kaggle competitions download -c quickdraw-doodle-recognition -f train_simplified.zip
unzip train_simplified.zip -d train
```

### Convert to tfrecords
```bash
cd <project_dir>/src/
python3 kaggle_store_tfrecords.py --store train
python3 kaggle_store_tfrecords.py --store test
```

## Use dataset in Tensorflow code
General use:
```python3
from sketch_io import train_data, valid_data, test_data, write_results

data_fn = train_data # Use training dataset, otherwise use valid_data/test_data
id_op, country_op, lengths_op, sketch_op, labels_op = \
    data_fn(batch_size, epochs, small, with_imgs=False, max_seqlen)

# Get additional image snapshots of the sketches. Images created live
id_op, country_op, lengths_op, sketch_op, num_images_op, images_op, image_indices_op, labels_op = \
    data_fn(batch_size, epochs, small, with_imgs=True, max_seqlen)
```

Example usage in SketchModels.py's `get_data` function (together with pre-processing of the sketches).

Illustration of class label prediction in SketchModel.py's `predict` function.

## Train models

Train the four individual classifiers by executing
```bash
python3 Sketch{CNN,LSTM,BLSTM,VRNN}.py --batch_size 32 --learning_rate 1e-4 --max_seqlen 150
# For more flag options check _init_flags function in the respective file

# Bare-bones training of CNN, allowing for higher batch sizes
python3 SketchCNNsimple.py --batch_size 128
```

## Sampling from trained VRNN network

Sampling new sketches through the VRNN network is most times not able to produce good-looking sketches. How to sample from the p-/q-distribution is shown in pSample.ipynb, qSample.ipynb and pSample_after_half.ipynp notebooks.

## KNN
To convert samples into the representation space and output neighbors, see sketch_create_features.ipynb, sketch_knn_parse.ipynb and sketch_knn_visualization.ipynb notebooks.



