from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from pkl import load_training
from pkl import load_testing
from pkl import write_predictions
import numpy as np

from CNN import cnn_model
from configuration import configure

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(2)


def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b


#_________________________________________MAIN_________________________________________
def main(unused_argv):

	x_train, y_train = load_training("../../data",True)
	config = configure()

	split = config.runCNN["split"]
	size = len(x_train)
	training_size = int(size - size*split)

	x_train, y_train = randomize(x_train,y_train)

	x_valid = x_train[training_size:size]
	y_valid = y_train[training_size:size]
	x_train = x_train[0:training_size]
	y_train = y_train[0:training_size]

	mnist_classifier = tf.estimator.Estimator(
		model_fn=cnn_model, model_dir=config.runCNN["model_dir"], config=tf.contrib.learn.RunConfig(save_checkpoints_steps=1000))

	if config.runCNN["training"]:

		epoch = config.runCNN["epochs"]

		for i in range(epoch):

			# Create the Estimator
			train_input_fn = tf.estimator.inputs.numpy_input_fn(
				x={"x": x_train},
				y=y_train,
				batch_size=config.runCNN["batch_size"],
				num_epochs=None,
				shuffle=True)
			mnist_classifier.train(
				input_fn=train_input_fn,
				steps=config.runCNN["steps"],
			)
			valid_input_fn = tf.estimator.inputs.numpy_input_fn(
				x={"x": x_valid},
				y=y_valid,
				num_epochs=1,
				shuffle=False)

			mnist_classifier.evaluate(input_fn=valid_input_fn)

	elif config.runCNN["evaluate"]:
		valid_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": x_valid},
			y=y_valid,
			num_epochs=1,
			shuffle=False)

		mnist_classifier.evaluate(input_fn=valid_input_fn)

	if config.runCNN["make_test_predictions"]:
		# Load test data
		x_test = load_testing("../../data")

		# Make Predictions
		predict_input_fn = tf.estimator.inputs.numpy_input_fn(
			  x={"x":x_test},
			  num_epochs=1,
			  shuffle=False)
		predictions = list(mnist_classifier.predict(input_fn=predict_input_fn))
		predicted_classes = [p["classes"] for p in predictions]

		# Print them to a file
		write_predictions("../../output", predicted_classes)


if __name__ == "__main__":
  tf.app.run()
