# Import the libraries (0.25 pt)
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from helper.helper import cross_validation


def predict(df):
    X = df[df.columns.drop(["Class"]).values]  # Features
    y = df.Class  # Target variable
    # Split the dataset into Training and Test groups (use 20-80 split, i.e. 20% of data will be used for the Test
    # group and 80% for training). (0.5 pt)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=16)
    # Perform feature scaling. (do not scale y – remember y=0/1 so it needs no scaling). (0.5 pt)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    # Train each of the above models and make predictions. Then compare the results by displaying the predicted
    # values of y next to the test values of y in a two-dimensional array. (2 pt)
    clf = svm.SVC().fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    print("Predict y: ")
    print(y_pred)
    print()  # Just for space
    print("Test y: ")
    print(y_test.to_numpy())
    cross_validation(svm.SVC(), X, y, y_test, y_pred)
    return confusion_matrix_evaluation(y_test, y_pred)


# Create and print the Confusion Matrix and the accuracy scores for each model (In this cAase is for Logistic
# Regression). (1 pt)
def confusion_matrix_evaluation(y_test, y_pred):
    from sklearn.metrics import classification_report
    target_names = ['Beginning', 'Malign']
    return classification_report(y_test, y_pred, target_names=target_names)
