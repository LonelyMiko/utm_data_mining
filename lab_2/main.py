import pandas as pd

from helper.helper import print_title
from models import DecisionTreeClassifierModel, LogisticRegressionModel, RandomForestClassifierModel, \
    KNeighborsClassifierModel, GaussianNaiveBayesModel, SupportVectorMachinesModel

if __name__ == '__main__':
    PATH = "../resources/Data1.csv"
    # Import the dataset (0.25 pt)
    df = pd.read_csv(PATH)
    print_title("LogisticRegressionModel")
    print(LogisticRegressionModel.predict(df))
    print_title("DecisionTreeClassifierModel")
    print(DecisionTreeClassifierModel.predict(df))

    print_title("RandomForestClassifierModel")
    print(RandomForestClassifierModel.predict(df))

    print_title("KNeighborsClassifierModel")
    print(KNeighborsClassifierModel.predict(df))

    print_title("GaussianNaiveBayesModel")
    print(GaussianNaiveBayesModel.predict(df))

    print_title("SupportVectorMachinesModel")
    print(SupportVectorMachinesModel.predict(df))
