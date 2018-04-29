import tensorflow as tf


class configure():
	# For runCNN.py
	runCNN = {
		"split": 0.30, # Split for validation set; 0.05 -> 2500 entries in validation set (this is not random, the 2500 will be the same)
		"model_dir": "./models/final5", # CNN checkpoints saved here. For different models, change this to a different number ex. CNN-2
		"log_dir": "./logs/final5",
		"epochs": 30,
		"steps": 500,
		"batch_size": 64,
		"training": False,
		"evaluate": False,
		"make_test_predictions": True
	}

	# For CNN.py
	CNN = {
		"optimizer": tf.train.AdamOptimizer(learning_rate=0.0009,beta1=0.8),
		"loss": tf.losses.softmax_cross_entropy,
		"conv_layers": [
			# Layer 1
			{"filters": 48,
			 "kernel_size": 5,
			 "kernel_initializer_stddev": 0.05,
			 "bias_initializer_min_max": [0.0,0.05],
			 "activation": tf.nn.relu,
			 # Pooling
			 "pool_size": 2,
			 "strides": 2,
			 },

			# Layer 2
			{"filters": 48,
			 "kernel_size": 5,
			 "kernel_initializer_stddev": 0.05,
			 "bias_initializer_min_max": [0.0, 0.05],
			 "activation": tf.nn.relu,
			 # Pooling
			 "pool_size": 2,
			 "strides": 2,
			 },

			# Layer 3
			{"filters": 48,
			 "kernel_size": 5,
			 "kernel_initializer_stddev": 0.05,
			 "bias_initializer_min_max": [0.0, 0.05],
			 "activation": tf.nn.relu,
			 # Pooling
			 "pool_size": 2,
			 "strides": 2,
			 },

			# Layer 4
			{"filters": 48,
			 "kernel_size": 5,
			 "kernel_initializer_stddev": 0.05,
			 "bias_initializer_min_max": [0.0, 0.05],
			 "activation": tf.nn.relu,
			 # Pooling
			 "pool_size": 2,
			 "strides": 2,
			 }
		],

		"dense_layers": [
		# 	Layer 1
			{"units": 1024,
			 "activation": tf.nn.relu,
			 "dropout_rate": 0.5
			},

			# 	Layer 2
			{"units": 512,
			 "activation": tf.nn.relu,
			 "dropout_rate": 0.5
			 }
		]

	}

