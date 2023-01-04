import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from Utils.utils import get_categorical_columns, get_numerical_columns

# local path to the CSV file with the data set to analise
DATASET_PATH = "dataSet/warranty_claims.csv"
# unnecessary column data
ERASE_COLUMNS = ["ID"]


# Loads a CSV into a data frame.
# Erases the unnecessary columns as in 'ID'.
def load_data_set(dataset_name, erase_columns):
    return pd.read_csv(filepath_or_buffer=dataset_name).drop(columns=erase_columns)


# replaces the values of a certain column from replacer[0] to replacer[-1]
def fix_values(data, column, replacer):
    return data[column].replace(replacer[0], replacer[-1])


# Splits the data frame into X input and Y output
def split_data_frame(dataframe, spliter):
    return dataframe.drop(spliter, axis='columns'), dataframe[spliter]


# makes and trains the data frame
def make_train_model(data_frame):
    # fixing values
    data_frame["Purpose"] = fix_values(data=data_frame, column='Purpose', replacer=['claim', 'Claim'])
    # splitting data frame into source and target
    x, y = split_data_frame(dataframe=data_frame, spliter='Fraud')

    # randomly separating data frame in train test model with 90:10 ratio
    source_train, source_test, target_train, target_test = train_test_model(x_data=x, y_data=y, train_ratio=0.9)

    # column transformers
    num_column_transformer = make_column_transformer((SimpleImputer(), get_numerical_columns(data_frame=data_frame)),
                                                     remainder='passthrough').set_output(transform="pandas")
    # fill missing values
    source_train = num_column_transformer.fit_transform(source_train)
    source_test = num_column_transformer.fit_transform(source_test)

    # standardize numerical data, all numeric except Fraud
    standardizer(source_train, get_numerical_columns(source_train))
    standardizer(source_test, get_numerical_columns(source_train))

    # read new table to get new columns names for categorical data
    cat_column_transformer = make_column_transformer(
        (OneHotEncoder(), get_categorical_columns(data_frame=source_train)),
        remainder='passthrough')

    # one hot encode
    source_train, source_test = one_hot_encoded(cat_column_transformer, source_train=source_train,
                                                source_test=source_test)
    return source_train, source_test, target_train, target_test


# Splits the dataframe into train/test with ratio
def train_test_model(x_data, y_data, train_ratio):
    return train_test_split(x_data, y_data, test_size=1 - train_ratio, train_size=train_ratio)


# one hot encodes all categorical columns and leaves the numerical untouched
def one_hot_encoded(data, source_train, source_test):
    return data.fit_transform(source_train), data.fit_transform(source_test)


# Standardizes the numerical values present in the numerical columns in the data_frame
def standardizer(data_frame, source_num_columns):
    scaler = StandardScaler().set_output(transform="pandas")
    data_frame[source_num_columns] = scaler.fit_transform(data_frame[source_num_columns])
    return data_frame


# classifies the source data with the target train pool
def predictor(source_train, source_test, target_train):
    model = MLPClassifier()
    model.fit(source_train, target_train)
    return model.predict(source_test)


# Classifies the overall solution using a confusion matrix
def classifier(target_test, target_predict):
    classification_rep = classification_report(target_test, target_predict, target_names=["No Fraud", "Fraud"])
    conf_matrix = confusion_matrix(target_test, target_predict)
    print(classification_rep)
    return classification_rep, conf_matrix


# Entry point
if __name__ == '__main__':
    # loading data set into pandas data frame
    data_set = load_data_set(dataset_name=DATASET_PATH, erase_columns=ERASE_COLUMNS)
    # making and training data frame
    x_train, x_test, y_train, y_test = make_train_model(data_frame=data_set)
    # classifying the train test model with source train and test and target train
    score = predictor(source_train=x_train, source_test=x_test, target_train=y_train)
    print("Score =", accuracy_score(y_test, score))
    # classifying overall solution with previous classification and target test values
    classifier(target_test=y_test, target_predict=score)
