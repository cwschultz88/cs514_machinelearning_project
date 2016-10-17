from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from data_extractor import TrainingDataExtractor

datasource = TrainingDataExtractor()
X_training, X_validation, y_training, y_validation = datasource.split_into_training_and_valdation()

classifiers = []
classifiers.append((RandomForestClassifier(n_estimators=300, max_depth=10, max_features=0.4, n_jobs=10, random_state=123), 'Random Forest Classifier 300 10 0.4'))
classifiers.append((RandomForestClassifier(n_estimators=300, max_depth=15, max_features=0.4, n_jobs=10, random_state=123), 'Random Forest Classifier 300 15 0.4'))
classifiers.append((RandomForestClassifier(n_estimators=300, max_depth=15, max_features=0.6, n_jobs=10, random_state=123), 'Random Forest Classifier 300 15 0.6'))
classifiers.append((RandomForestClassifier(n_estimators=400, max_depth=15, max_features=0.5, n_jobs=10, random_state=123), 'Random Forest Classifier 400 15 0.5'))
classifiers.append((RandomForestClassifier(n_estimators=500, max_depth=15, max_features=0.5, n_jobs=10, random_state=123), 'Random Forest Classifier 500 15 0.5'))
classifiers.append((RandomForestClassifier(n_estimators=500, max_depth=10, max_features=0.4, n_jobs=10, random_state=123), 'Random Forest Classifier 500 10 0.4'))


for classifier, classifier_str in classifiers:
    classifier.fit(X_training, y_training)

    y_predicted = classifier.predict(X_validation)
    predict_probabilities = classifier.predict_proba(X_validation)
    positive_probabilities = [predict_probability[1] for predict_probability in predict_probabilities]

    print classifier_str
    print "ROC auc score: " + str(roc_auc_score(y_validation, positive_probabilities))
    print "Accuracy: " + str(accuracy_score(y_validation, y_predicted))
    print ""
