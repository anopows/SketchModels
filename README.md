# Representation Learning for Sketches

## About
In this project, a representation space is learned for sketch classification and providing suggestions of similar sketches on the [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset).

For instructions of code implemenation please visit the [Instructions README](/src/README.md).

Sample sketches of the dataset:
![Quick, Draw!](/images/quickdraw.png?raw=true)

## Models 
Four different models are implemented on both temporal and image data: LSTM, BLSTM, ResNet-v2, VRNN. 

![Representations](/images/representations.png?raw=true)

A model with combined mode of representation can be found in this repository: [Combined Representations](https://github.com/anopows/Combined-SketchModels)

### VRNN Implementation
Overview of VRNN architecture: 
![VRNN-Implementation](/images/vrnn.png?raw=true)

The learned repesentation space is a merged vector consisting of h and z states: 
![VRNN-representation](/images/vrnn-representation.png?raw=true)

To improve classification accuracy, the VRNN losses (KL-divergence, loglikelihood loss) and classification loss were added together to form a combined loss function.

## Results
### Classification
For a given batch size VRNN performs best in terms of classificaiton accuracy (trained with batch size 32):

![Results](/images/classification.png?raw=true)

### Suggestions
Looking up the closest neighbors in the learned representation space, the VRNN network generally has neighbors with closer matching styles. e.g. wheels drawn with two circles: 
![Neighbors](/images/neighbors.png?raw=true)

### Batch size
It is to be noted that the batch size affects the training performance significantly. This limits the implementations of larger models, as they don't fit into GPU memory when used with high batch size: 

![BatchSize](/images/batch_size.png?raw=true)

### Possible explanation of batch size effect
One possible reason is that a batch with a smaller batch size has higher likelihood to include mostly faulty sketches.

![Faulty](/images/faulty.png?raw=true)
