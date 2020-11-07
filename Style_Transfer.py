import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


content_path = 'Content.jpg'
style_path= 'Style.jpg'

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0) 
    ### remove dim of size 1 

  plt.imshow(image)
  if title:
    plt.title(title)


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  ### this one read file in the form of A Tensor of type string.
  img = tf.image.decode_image(img, channels=3)
  #Detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the appropriate operation to convert the input bytes string into a Tensor of type dtype.
  img = tf.image.convert_image_dtype(img, tf.float32)

  #rescaling the image
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  #changing the type of shape to do floting operations on them
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img
  