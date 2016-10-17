from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from data_extractor import TrainingDataExtractor

datasource = TrainingDataExtractor()
features, labels = datasource.all_data()

roc_scores = []
kf = KFold(len(features), 6, shuffle=False, random_state=None)
for train_indices, test_indices in kf:
    X_train, X_test = [features[train_index] for train_index in train_indices], [features[test_index] for test_index in test_indices]
    y_train, y_test = [labels[train_index] for train_index in train_indices], [labels[test_index] for test_index in test_indices]

    classifier = RandomForestClassifier(n_estimators=500, max_depth=10, max_features=0.4, n_jobs=2)
    classifier.fit(X_train, y_train)

    predict_probabilities = classifier.predict_proba(X_test)
    positive_probabilities = [predict_probability[1] for predict_probability in predict_probabilities]
    roc_scores.append(roc_auc_score(y_test, positive_probabilities))

print "Final Model ROC Score: " + str(sum(roc_scores)/float(len(roc_scores)))
