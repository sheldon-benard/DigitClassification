from preprocessing import threshold_normalize
import numpy as np
import _pickle as pickle
import os
import sys

def pre_and_pickle(dataDir, filename_x, transform,filename_y=None):
	csv = dataDir + "/" + filename_x + ".csv"
	pkl = dataDir + "/" + filename_x + ".pkl"
	elastic = dataDir + "/" + filename_x + "_elastic.pkl"
	rot = dataDir + "/" + filename_x + "_rot.pkl"

	if os.path.isfile(csv):
		print("Read: " + csv)
		x_data = np.loadtxt(csv, delimiter=",")
		x_data = x_data.reshape(-1,64,64)
		if transform:
			# x_data,elastic_data,rot_data = threshold_normalize(x_data,transform)
			x_data,elastic_data = threshold_normalize(x_data,transform)

			print("Write: " + pkl)
			with open(pkl,'wb') as f:
				pickle.dump(x_data, f)
			with open(elastic,'wb') as f:
				pickle.dump(elastic_data,f)
			# with open(rot,'wb') as f:
			# 	pickle.dump(rot_data,f)

		else:
			x_data = threshold_normalize(x_data,transform)

			print("Write: " + pkl)
			with open(pkl,'wb') as f:
				pickle.dump(x_data, f)

	else:
		print("No csv file located at: " + csv)

	if not filename_y == None:
		csv_y = dataDir + "/" + filename_y + ".csv"
		pkl_y = dataDir + "/" + filename_y + ".pkl"

		if os.path.isfile(csv_y):
			print("Read: " + csv_y)
			y_data = np.loadtxt(csv_y, delimiter=",").reshape(-1)

			if transform:
				# copy = np.append(y_data, y_data, axis=0)
				y_data = np.append(y_data, y_data, axis=0)

			print("Write: " + pkl_y)
			with open(pkl_y,'wb') as f:
				pickle.dump(y_data, f)

		else:
			print("No csv file located at: " + csv_y)

	print("Done")

def load_training(dataDir, elastic, filename_x=None, filename_y=None):
	print("Loading training data")
	pkl_x = None
	pkl_y = None


	if filename_x == None and filename_y == None:
		pkl_x = dataDir + "/train_x.pkl"
		pkl_y = dataDir + "/train_y.pkl"

	else:
		pkl_x = dataDir + "/" + filename_x + ".pkl"
		pkl_y = dataDir + "/" + filename_y + ".pkl"


	x_in = open(pkl_x,'rb')
	y_in = open(pkl_y,'rb')

	x_data = pickle.load(x_in)
	y_data = pickle.load(y_in)

	x_in.close()
	y_in.close()

	print("Done")

	if elastic:
		print("Get elastic")
		pkl_elastic = dataDir + "/train_x_elastic.pkl"

		x_in = open(pkl_elastic, 'rb')
		x_data_elastic = pickle.load(x_in)
		x_in.close()

		x_data = np.asarray(x_data, dtype=np.float32)
		x_data_elastic = np.asarray(x_data_elastic, dtype=np.float32)

		x_data = np.append(x_data, x_data_elastic, axis=0)
		print(x_data.shape)
		print(y_data.shape)

	else:
		x_data = np.asarray(x_data, dtype=np.float32)

	return x_data, np.asarray(y_data, dtype=np.int32)


def load_testing(dataDir, filename=None):
	print("Loading test data")
	pkl_x = None

	if filename == None:
		pkl_x = dataDir + "/test_x.pkl"

	else:
		pkl_x = dataDir + "/" + filename + ".pkl"

	x_in = open(pkl_x,'rb')
	x_data = pickle.load(x_in)
	x_in.close()

	print("Done")

	return np.asarray(x_data,dtype=np.float32)


def write_predictions(outputDir, predictions, name=None):
	fileName = None
	if name == None:
		fileName = "test_predictions.csv"
	else:
		fileName = name

	print("Write predictions")
	with open(outputDir + "/" + fileName, 'w') as f:
		ID = 0

		f.write("Id,Label\n")
		for prediction in predictions:
			f.write(str(ID) + "," + str(prediction) + "\n")
			ID += 1

	print("Done")


if __name__ == "__main__":
	# try:
	if sys.argv[1] == "CNN":
		pre_and_pickle("../../data", "test_x", False)
	elif sys.argv[1] == "all":
		pre_and_pickle("../../data", "test_x", False)
		pre_and_pickle("../../data", "train_x", False, "train_y")
	else:
		print("Invalid argument")
	# except:
	# 	print("No argument provided")





