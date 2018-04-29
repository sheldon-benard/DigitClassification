import numpy as np
# Linear SVC/SVM
from sklearn.svm import LinearSVC
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import _pickle as pickle



def generateSVMResult():
	clf = LinearSVC()

	dataDir = "../../data"
	outputDir = "../../output"
	pkl_x = dataDir + "/train_x.pkl"
	pkl_y = dataDir + "/train_y.pkl"
	x_in = open(pkl_x,'rb')
	y_in = open(pkl_y,'rb')

	x_train = pickle.load(x_in)
	y_train = pickle.load(y_in)

	x_in.close()
	y_in.close()


	pkl_test_x = dataDir + "/test_x.pkl"
	x_in = open(pkl_test_x,'rb')
	x_test = pickle.load(x_in)
	x_in.close()

	x_test = np.asarray(x_test,dtype=np.float32).reshape(-1,64*64)

	x_train = np.asarray(x_train,dtype=np.float32).reshape(-1,64*64)
	y_train = np.asarray(y_train, dtype=np.int32).reshape(-1)[0:50000]

	size = len(x_train)
	training_size = int(size - size*0.30)

	train_test_fold = np.full(training_size, 0)
	valid_test_fold = np.full(size-training_size, 1)
	test_fold = np.append(train_test_fold, valid_test_fold, axis=0)

	ps = PredefinedSplit(test_fold=test_fold)

	# We have 4096 features and > 10,000 inputs, so solve primal
	tuned_parameters = [
		{'penalty': ['l2'],
		 'loss':['squared_hinge'],
		 'dual':[False],
		'C':[0.01],
		}]

	clf = GridSearchCV(clf, tuned_parameters, cv=ps, refit=True, verbose=10)
	clf.fit(x_train, y_train)

	print(clf.best_estimator_)

	print(clf.best_score_)

	y_pred = clf.predict(x_test)

	print("Write predictions")
	with open(outputDir + "/" + "test_predictions_svm.csv", 'w') as f:
		ID = 0

		f.write("Id,Label\n")
		for prediction in y_pred:
			f.write(str(ID) + "," + str(prediction) + "\n")
			ID += 1

	print("Done")


def main():
	generateSVMResult()


if __name__ == "__main__": main()


