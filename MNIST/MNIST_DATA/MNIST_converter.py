import gzip
import os
from urllib.request import urlretrieve
import numpy as np
import pickle

# Download dataset from https://deepai.org/dataset/mnist
# Reference: https://mattpetersen.github.io/load-mnist-with-numpy


with gzip.open('./mnist/train-images-idx3-ubyte.gz') as f:
    pixels = np.frombuffer(f.read(), 'B', offset=16)
    train_image = pixels.reshape(-1, 784)

with gzip.open('./mnist/t10k-images-idx3-ubyte.gz') as f:
    pixels = np.frombuffer(f.read(), 'B', offset=16)
    test_image = pixels.reshape(-1, 784)

with gzip.open('./mnist/train-labels-idx1-ubyte.gz') as f:
    train_labels = np.frombuffer(f.read(), 'B', offset=8)

with gzip.open('./mnist/t10k-labels-idx1-ubyte.gz') as f:
    test_labels = np.frombuffer(f.read(), 'B', offset=8)


save_file = open('mnist.pkl', 'wb')
save_content = {
    'training_images': train_image,
    'training_labels': train_labels,
    'test_images': test_image,
    'test_labels': test_labels,
}
pickle.dump(save_content, save_file)
save_file.close()




print('\033[91mFINISHED\033[0m')


