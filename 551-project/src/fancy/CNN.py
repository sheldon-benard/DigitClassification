from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from configuration import configure

config = configure().CNN

def cnn_model(features, labels, mode):

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

    numConvLayers = len(config["conv_layers"])

    conv = []
    pool = []
    normal = []

    for i in range(numConvLayers):
        inp = None
        layer = config["conv_layers"][i]

        if i == 0:
            inp  = input_layer
        else:
            inp = normal[i-1]

        conv.append(
            tf.layers.conv2d(
                inputs=inp,
                filters=layer["filters"],
                kernel_size=[layer["kernel_size"], layer["kernel_size"]],
                padding="same",
                kernel_initializer=tf.truncated_normal_initializer(stddev=layer["kernel_initializer_stddev"]),
                bias_initializer=tf.random_uniform_initializer(minval=layer["bias_initializer_min_max"][0], maxval=layer["bias_initializer_min_max"][1]),
                activation=layer["activation"]
            )
        )

        pool.append(
            tf.layers.max_pooling2d(inputs=conv[i], pool_size=[layer["pool_size"], layer["pool_size"]], strides=layer["strides"])
        )

        # No normal for last layer
        if i != (numConvLayers - 1):
            normal.append(
                tf.layers.batch_normalization(inputs=pool[i])
            )

    # Flatten tensor into a batch of vectors; do this dynamically
    pool_flat = tf.contrib.layers.flatten(pool[-1])

    numDenseLayers = len(config["dense_layers"])
    dense = []
    dropout = []

    for j in range(numDenseLayers):
        inp = None
        layer = config["dense_layers"][j]

        if j == 0:
            inp = pool_flat
        else:
            inp = dropout[j-1]

        dense.append(
            tf.layers.dense(inputs=inp, units=layer["units"], activation=layer["activation"])
        )

        dropout.append(
            tf.layers.dropout(inputs=dense[j], rate=layer["dropout_rate"], training=mode == tf.estimator.ModeKeys.TRAIN)
        )

    #  Output Layer
    logits = tf.layers.dense(inputs=dropout[-1], units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = config["loss"](
      onehot_labels=onehot_labels, logits=logits)

    accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])

    # tf.summary.scalar('loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('accuracy', accuracy[1])
        optimizer = config["optimizer"]
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": accuracy}

    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
