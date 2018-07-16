# Image-Captioning
Caption an image using a deep neural network. It uses a model created by VGG(Visual Geometric Group) which is a combination of a 16- layered and a 19-layered Deep Neural Network to extract features from images. And captions these images using our Deep Neural Network Model.


# Dataset
I used a dataset of Flickr-8K Dataset of images and their labelled captions for training.
The dataset can be found [here](https://forms.illinois.edu/sec/1713398).
Fill in the form for requesting the data.

Within a short time, you will receive an email that contains links to two files:-
1) Flickr8k_Dataset.zip (1 GB) All photographs.
2) Flickr8k_text.zip (2.2 MB) All text descriptions for photographs.


# Requirements

python      3.5 or more
keras       2.1.0
tensorflow  1.8.0
nltk        3.2.4
numpy       1.13.3
Pillow      4.0.0
pickle
You also need to install pydot using pip install pydot.
Also you need to install GraphViz which can be found [here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html).

# Model used for extracting features.

There are many models to choose from. In this case, we will use the Oxford Visual Geometry Group, or VGG, model that won the ImageNet competition in 2014. [Read More](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)

# Model used for generating captions.

1) Sequence Processor: This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer.
2) Decoder: Both the feature extractor and sequence processor output a fixed-length vector. These are merged together and processed by a Dense layer to make a final prediction.
The Photo Feature Extractor model expects input photo features to be a vector of 4,096 elements. These are processed by a Dense layer to produce a 256 element representation of the photo.

The Sequence Processor model expects input sequences with a pre-defined length (40 words) which are fed into an Embedding layer that uses a mask to ignore padded values. This is followed by an LSTM layer with 256 memory units.

Both the input models produce a 256 element vector. Further, both input models use regularization in the form of 50% dropout. This is to reduce overfitting the training dataset, as this model configuration learns very fast.

The Decoder model merges the vectors from both input models using an addition operation. This is then fed to a Dense 256 neuron layer and then to a final output Dense layer that makes a softmax prediction over the entire output vocabulary for the next word in the sequence.

# Research Papers referred.
1) [Where to put the Image in an Image Caption Generator](https://arxiv.org/abs/1703.09137)
2) [What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?](https://arxiv.org/abs/1708.02043)
