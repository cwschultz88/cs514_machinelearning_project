import random
import sklearn.cross_validation
import pandas as pd
import numpy as np

class TrainingDataExtractor:
    def __init__(self, training_file_name="train.csv"):
        '''
        starting point:
        https://www.kaggle.com/mordock/santander-customer-satisfaction/0-834842-reduced-to-100-features/code
        '''
        with open(training_file_name, 'rb') as f:
            trainDF = pd.read_csv(training_file_name)

            # remove constant columns
            colsToRemove = []
            for col in trainDF.columns:
                if trainDF[col].std() == 0:
                    colsToRemove.append(col)

            trainDF.drop(colsToRemove, axis=1, inplace=True)

            # remove duplicate columns
            colsToRemove = []
            columns = trainDF.columns
            for i in range(len(columns)-1):
                v = trainDF[columns[i]].values
                for j in range(i+1, len(columns)):
                    if np.array_equal(v, trainDF[columns[j]].values):
                        colsToRemove.append(columns[j])
            trainDF.drop(colsToRemove, axis=1, inplace=True)

            labels = trainDF.TARGET.values.tolist()
            features = trainDF.drop(['ID','TARGET'], axis=1).values.tolist()

            self.features = features
            self.labels = labels

    def all_data(self):
        return self.features, self.labels

    def split_into_training_and_valdation(self, split_amount=0.25):
        '''
        returns 4 - tuple  (training_features, validation_features, training_labels, validation_labels)
        '''
        return sklearn.cross_validation.train_test_split(self.features, self.labels, test_size=split_amount)

