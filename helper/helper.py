import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# For each model, use the K-fold cross-validation (use K=10; in python - cv=10).
# Print the mean of all 10 accuracy scores for each model and their standard deviations. (1 pt)
def cross_validation(model, X, y, y_test, y_predict):
    # define cross-validation method to use
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    # use k-fold CV to evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',
                             cv=cv, n_jobs=-1)
    print('\n\nCross-Validation Standard Deviations: %.3f' % np.std(scores))
    cross_accuracy = accuracy_score(y_test, y_predict)
    print("Cross-Validation Accuracy Score %.3f" % cross_accuracy)
    return cross_accuracy


# For each model compare the accuracy scores computed using cross-validation in (7) versus when using only one test
# set in (6). Are the mean accuracy scores from cross-validation higher or lower in comparison to the corresponding
# scores in (6)? Did you expect them to be higher or lower? Why? (1.5 pt)
def compare_accuracy(cross_accuracy, f1_cccuracy):
    return cross_accuracy - f1_cccuracy


def print_title(title):
    print()
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
    print("                               " + title + "                               ")
    print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
    print()

