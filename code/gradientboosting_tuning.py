from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from data_extractor import TrainingDataExtractor

datasource = TrainingDataExtractor()
X_training, X_validation, y_training, y_validation = datasource.split_into_training_and_valdation()

classifiers = []
classifiers.append((GradientBoostingClassifier(n_estimators=100), 'Gradient Boosting Classifier 100'))
classifiers.append((GradientBoostingClassifier(n_estimators=300), 'Gradient Boosting Classifier 300'))
classifiers.append((GradientBoostingClassifier(n_estimators=500), 'Gradient Boosting Classifier 500'))
classifiers.append((GradientBoostingClassifier(n_estimators=750), 'Gradient Boosting Classifier 750'))


for classifier, classifier_str in classifiers:
    classifier.fit(X_training, y_training)

    y_predicted = classifier.predict(X_validation)
    predict_probabilities = classifier.predict_proba(X_validation)
    positive_probabilities = [predict_probability[1] for predict_probability in predict_probabilities]

    print classifier_str
    print "ROC auc score: " + str(roc_auc_score(y_validation, positive_probabilities))
    print "Accuracy: " + str(accuracy_score(y_validation, y_predicted))
    print ""
