from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from data_extractor import TrainingDataExtractor

datasource = TrainingDataExtractor()
X_training, X_validation, y_training, y_validation = datasource.split_into_training_and_valdation()

classifiers = []
classifiers.append((GradientBoostingClassifier(), 'Gradient Boosting Classifier'))
classifiers.append((RandomForestClassifier(), 'Random Forest Classifier'))
classifiers.append((AdaBoostClassifier(), 'AdaBoost Classifier'))
classifiers.append((LogisticRegression(), 'Logisitic Classifier'))


for classifier, classifier_str in classifiers:
    classifier.fit(X_training, y_training)

    y_predicted = classifier.predict(X_validation)
    predict_probabilities = classifier.predict_proba(X_validation)
    positive_probabilities = [predict_probability[1] for predict_probability in predict_probabilities]

    print classifier_str
    print "ROC auc score: " + str(roc_auc_score(y_validation, positive_probabilities))
    print "Accuracy: " + str(accuracy_score(y_validation, y_predicted))
    print ""
