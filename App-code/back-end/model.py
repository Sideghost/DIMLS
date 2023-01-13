import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

from Utils.utils import get_categorical_columns, get_numerical_columns

# local path to the CSV file with the data set to analise
DATASET_PATH = "data/warranty_claims.csv"
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

    # pipelines
    num_pipeline = Pipeline([('SimpleImputer', SimpleImputer()), ('standardizer', StandardScaler())])
    cat_pipeline = Pipeline([('one_hot_encoded', OneHotEncoder())])
    preprocessor = ColumnTransformer(transformers=[('number', num_pipeline, get_numerical_columns(data_frame)), ('categorical', cat_pipeline, get_categorical_columns(data_frame))])

    return source_train, source_test, target_train, target_test, preprocessor


# Splits the dataframe into train/test with ratio
def train_test_model(x_data, y_data, train_ratio):
    return train_test_split(x_data, y_data, test_size=1 - train_ratio, train_size=train_ratio)


# classifies the source data with the target train pool
def predictor(source_train, source_test, target_train):
    # lr = LogisticRegression()
    mpl = MLPClassifier(random_state=1, max_iter=300)
    return mpl.fit(source_train, target_train), mpl.predict(source_test), mpl


# Classifies the overall solution using a confusion matrix
def classifier(target_test, target_predict):
    return classification_report(target_test, target_predict, target_names=["No Fraud", "Fraud"])


# Entry point
if __name__ == '__main__':
    # loading data set into pandas data frame
    data_set = load_data_set(dataset_name=DATASET_PATH, erase_columns=ERASE_COLUMNS)
    # making and training data frame
    x_train, x_test, y_train, y_test, preprocessor = make_train_model(data_frame=data_set)
    # classifying the train test model with source train and test and target train
    new_pipe = Pipeline([("processor", preprocessor), ("classifier", MLPClassifier())])
    new_pipe.fit(x_train, y_train)
    # classifying overall solution with previous classification and target test values
    print(classifier(target_test=y_test, target_predict=new_pipe.predict(x_test)))

    model_file_w = open("model.plk", "wb")
    pickle.dump(new_pipe, model_file_w)
    model_file_w.close()
    print("Model dumped!")
    model_file_r = open("model.plk", "rb")
    model_file = pickle.load(model_file_r)
    model_file_r.close()
