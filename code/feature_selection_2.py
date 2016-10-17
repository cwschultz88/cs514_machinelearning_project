from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFpr
from sklearn.cross_validation import KFold
from data_extractor import TrainingDataExtractor
from sklearn.metrics import roc_auc_score

datasource = TrainingDataExtractor()
features, labels = datasource.all_data()

number_of_features_to_reduce_to = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

for reduce_down_to in number_of_features_to_reduce_to:
    print 'reducing down to ' + str(reduce_down_to)
    estimator = LogisticRegression()
    selector = SelectFpr(alpha=reduce_down_to)

    selector.fit(features, labels)

    print 'performing 6-fold cross-validation'
    kf = KFold(len(features), 6, shuffle=False, random_state=None)
    roc_scores = []
    for train_indices, test_indices in kf:
        X_train, X_test = [features[train_index] for train_index in train_indices], [features[test_index] for test_index in test_indices]
        y_train, y_test = [labels[train_index] for train_index in train_indices], [labels[test_index] for test_index in test_indices]

        test_model = LogisticRegression()

        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        test_model.fit(X_train, y_train)

        y_predicted = test_model.predict(X_test)
        predict_probabilities = test_model.predict_proba(X_test)
        positive_probabilities = [predict_probability[1] for predict_probability in predict_probabilities]
        roc_scores.append(roc_auc_score(y_test, positive_probabilities))
    print "Features left: " + str(len(X_train[0])) + " out of " + str(len(features[0]))
    print "ROC auc score: " + str(sum(roc_scores)/float(len(roc_scores)))
    print ""
