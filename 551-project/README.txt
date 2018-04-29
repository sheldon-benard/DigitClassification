# 551 Final Project

Sheldon Benard (260618386)
Bing'er Jiang (260668025
Elias Stengel-Eskin (260609642)

1. Navigate to the project code root folder (folder called 551-project in the zipped file)

2. (Optional) Setup a virtualenv - so that the pip install is local

	'virtualenv .'

**We used python v3 for the project (ex. Python 3.6.3), so make sure you use python v3 to run the project. To check the version:

	'python --version' 

3. (Optional) If you setup a virtualenv, run:

	'source bin/activate'

4. Now, install dependencies

	'pip install -r requirements.txt'

5. Now, download data and preprocess

	'python get_started.py'

6. You are good to run the classifiers:

a. Linear SVM:

From the project code root folder

	'cd src/baselines'

Run linearSVM

	'python lin_svm.py'

This will generate "test_predictions_svm.csv" in the project code root output folder (551-project/output)


b. Run CNN Ensemble:

From the project code root folder

	'cd src/fancy'

Run ensemble:

	'python runCNNEnsemble.py'

Get majority vote:

	'python majorityVoting.py'

This will generate several prediction files in the 551-project/output folder. The prediction file that we desire (the majority voting prediction) is called "test_predictions.csv"


c. Run FFNN:

From the project code root folder

	'cd src/ffnn'

Run FFNN:

	'python mpl.py'

This will train the FFNN and generate "test_predictions_ffnn.csv" in the 551-project/output folder

