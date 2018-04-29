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

config = configure()

#_________________________________________MAIN_________________________________________
def main(unused_argv):

	if config.runCNN["make_test_predictions"]:

		x_test = load_testing("../../data")
		models = [
		{
			"name": "final","checkpoints": ["model.ckpt-10201","model.ckpt-10301","model.ckpt-10401","model.ckpt-10500","model.ckpt-10501"]
		},
		{
			"name": "final2","checkpoints": ["model.ckpt-9601","model.ckpt-9701","model.ckpt-9801","model.ckpt-9901","model.ckpt-10000"]
		},
		{
			"name": "final3","checkpoints": ["model.ckpt-9601","model.ckpt-9701","model.ckpt-9801","model.ckpt-9901","model.ckpt-10000"]
		},
		{
			"name": "final4","checkpoints": ["model.ckpt-19000","model.ckpt-19001","model.ckpt-19500","model.ckpt-19501","model.ckpt-20000"]
		},
		{
			"name": "final5","checkpoints": ["model.ckpt-14000","model.ckpt-14001","model.ckpt-14500","model.ckpt-14501","model.ckpt-15000"]
		}
		 ]

		for model in models:
			name = "./models/" + model["name"]

			mnist_classifier = tf.estimator.Estimator(
				model_fn=cnn_model, model_dir=name)

			for checkpoint in model["checkpoints"]:

				# Make Predictions
				predict_input_fn = tf.estimator.inputs.numpy_input_fn(
					  x={"x":x_test},
					  num_epochs=1,
					  shuffle=False,)

				check = name + "/" + checkpoint

				predictions = list(mnist_classifier.predict(input_fn=predict_input_fn,checkpoint_path=check))
				predicted_classes = [p["classes"] for p in predictions]

				# Print them to a file
				write_predictions("../../output", predicted_classes, model["name"] + checkpoint + ".csv")


if __name__ == "__main__":
  tf.app.run()
