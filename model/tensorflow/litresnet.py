from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
#learning_rate = 0.000001
#dropout = 0.5 # Dropout, probability to keep units
#training_iters = 6001
#step_display = 50
#step_save = 6000
#path_save = 'model_alexnet_bn_no_data_aug_lr000001_v3'
#start_from = 'model_alexnet_bn_no_data_aug_v2-6000'

##################################

### moved training params to the __main__ func

##################################

###################### resnet below ####################################


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   data_format):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

  return tf.identity(inputs, –)


def upsample_layer(input_img, factor):
  number_of_classes = input_img.shape[2]
  new_height = input_img.shape[0] * factor
  new_width = input_img.shape[1] * factor

  expanded_img = tf.expand_dims(input_img, axis=0)

  upsample_filt_pl = tf.placeholder(tf.float32)
  logits_pl = tf.placeholder(tf.float32)
  upsample_filter_np = bilinear_upsample_weights(factor,
                                        number_of_classes)

  res = tf.nn.conv2d_transpose(logits_pl, upsample_filt_pl,
                        output_shape=[1, new_height, new_width, number_of_classes],
                        strides=[1, factor, factor, 1])
   return res


def imagenet_resnet_v2_generator(block_fn, layers, num_classes,
                                 data_format=None):
  """Generator for ImageNet ResNet v2 models.

  Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  """
  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)


    inputs = upsample_layer(input_img, factor)

    inputs = upsample_layer(input_img, factor)

    inputs = upsample_layer(inputs_img, factor)


    # inputs = tf.layers.average_pooling2d(
    #     inputs=inputs, pool_size=7, strides=1, padding='VALID',
    #     data_format=data_format)
    # inputs = tf.identity(inputs, 'final_avg_pool')
    # inputs = tf.reshape(inputs,
    #                     [-1, 512 if block_fn is building_block else 2048])
    # inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    # inputs = tf.identity(inputs, 'final_dense')
    return inputs

  return model


def imagenet_resnet_v2(resnet_size, num_classes, data_format=None):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': building_block, 'layers': [2, 2, 2, 2]},
      34: {'block': building_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_size not in model_params:
    raise ValueError('Not a valid resnet_size:', resnet_size)

  params = model_params[resnet_size]
  return imagenet_resnet_v2_generator(
      params['block'], params['layers'], num_classes, data_format)

  return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)





###################### resnet above ##################################

def main():

    # Construct dataloader
    opt_data_train = {
        #'data_h5': 'miniplaces_256_train.h5',
        'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
        'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False
        }
    opt_data_val = {
        #'data_h5': 'miniplaces_256_val.h5',
        'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
        'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': True
        }

    loader_train = DataLoaderDisk(is_train=True, **opt_data_train)
    loader_val = DataLoaderDisk(**opt_data_val)
    #loader_train = DataLoaderH5(**opt_data_train)
    #loader_val = DataLoaderH5(**opt_data_val)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
    y = tf.placeholder(tf.int64, None)
    keep_dropout = tf.placeholder(tf.float32)
    train_phase = tf.placeholder(tf.bool)

    # Construct model
    model= imagenet_resnet_v2(RESNET_SIZE, 100, data_format=None)
    output= model(x,True)

    #print(output)

    # Define loss and optimizer
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    #train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Evaluate model
    #accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
    #accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

    # define initialization
    init = tf.global_variables_initializer()

    # define saver
    saver = tf.train.Saver()

    # define summary writer
    #writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

    # Launch the graph
    with tf.Session() as sess:
        # Initialization
        if len(start_from)>1:
            saver.restore(sess, start_from)
        else:
            sess.run(init)
        
        step = 0

        while step < training_iters:
            # Load a batch of training data
            images_batch, labels_batch = loader_train.next_batch(batch_size)
            
            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                # Calculate batch loss and accuracy on training set
                #l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
                out1 = sess.run([output], feed_dict={x:images_batch, y:labels_batch, keep_dropout: 1., train_phase:False}
                print("-Iter " + str(step) + ", Training Loss= " + \
                      "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.4f}".format(acc1) + ", Top5 = " + \
                      "{:.4f}".format(acc5))

                # Calculate batch loss and accuracy on validation set
                images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
                print("-Iter " + str(step) + ", Validation Loss= " + \
                      "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.4f}".format(acc1) + ", Top5 = " + \
                      "{:.4f}".format(acc5))
            
            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
            
            step += 1
            
            # Save model
            if step % step_save == 0:
                saver.save(sess, path_save, global_step=step)
                print("Model saved at Iter %d !" %(step))
            
        print("Optimization Finished!")


        # Evaluate on the whole validation set
        print('Evaluation on the whole validation set...')
        num_batch = loader_val.size()//batch_size
        acc1_total = 0.
        acc5_total = 0.
        loader_val.reset()
        for i in range(num_batch):
            images_batch, labels_batch = loader_val.next_batch(batch_size)    
            acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
            acc1_total += acc1
            acc5_total += acc5
            print("Validation Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

        acc1_total /= num_batch
        acc5_total /= num_batch
        print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))


# def main_test():

#     # Construct dataloader
#     opt_data_test = {
#         #'data_h5': 'miniplaces_256_train.h5',
#         'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
#         'data_list': '../../data/test.txt', # MODIFY PATH ACCORDINGLY
#         'load_size': load_size,
#         'fine_size': fine_size,
#         'data_mean': data_mean,
#         'randomize': False
#         }

#     loader_test = DataLoaderDiskTest(is_train=False, **opt_data_test)

#     # tf Graph input
#     x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
#     keep_dropout = tf.placeholder(tf.float32)
#     train_phase = tf.placeholder(tf.bool)
#     # Construct model
#     model= imagenet_resnet_v2(RESNET_SIZE, 100, data_format=None)
#     logits= model(x,True)
#     _, top5_preds = tf.nn.top_k(logits, 5)
#     # define initialization
#     init = tf.global_variables_initializer()
#     # define saver
#     saver = tf.train.Saver()
#     # Launch the graph
#     out_preds = []
#     num_batch = loader_test.size()//batch_size
#     with tf.Session() as sess:
#         # Initialization
#         saver.restore(sess, start_from)
#         # Evaluate on the whole validation set
#         loader_test.reset()
#         for i in range(num_batch):
#             images_batch = loader_test.next_batch(batch_size)    
#             top5s = sess.run([top5_preds], feed_dict={x: images_batch, keep_dropout: 1., train_phase: False})[0]
#             out_preds += top5s.tolist()

#     for i in range(10000):
#         num_str = str(i+1)
#         while len(num_str) < 8: num_str = '0' + num_str
#         preds_str = ' '.join([str(elem) for elem in out_preds[i]])
#         print('test/' + num_str + '.jpg ' + preds_str)

if __name__ == '__main__':
    RESNET_SIZE = 18 # 34
    is_test = False
    if not is_test:
        batch_size = 64
        learning_rate = 0.00001
        dropout = 0.50 # Dropout, probability to keep units
        training_iters = 1501
        step_display = 50
        step_save = 1500
        path_save = 'model_color_v1'
        start_from =''

        main()

    if is_test:
        batch_size = 100
        learning_rate = 0.00001
        dropout = 1.0 # Dropout, probability to keep units
        training_iters = 15000
        step_display = 50
        step_save = 5000
        path_save = 'model_testing'
        start_from = 'model_resnet-decrease-lr-more-iterations-data_aug_v3-1500'

        main_test()









