import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Utils.utils import get_categorical_columns, get_numerical_columns

# local path to the CSV file with the data set to analise
DATASET_PATH = "dataSet/warranty_claims.csv"
# unnecessary column data
ERASE_COLUMNS = ["ID"]


# Loads a CSV into a data frame.
# Erases the unnecessary columns as in 'ID'.
def load_data_set(dataset_name, erase_columns):
    return pd.read_csv(filepath_or_buffer=dataset_name).drop(columns=erase_columns)


# substitutes NaN values with mean value for categorical and numerical values
def treat_values(data_frame):
    for i in get_numerical_columns(data_frame=data_frame):
        data_frame[i] = data_frame[i].fillna(data_frame[i].mean())  # SimpleImputer
    for i in get_categorical_columns(data_frame=data_frame):
        data_frame[i] = data_frame[i].fillna(data_frame[i].mode()[0])
    return data_frame


# Splits the data frame into X input and Y output
def split_data_frame(dataframe, spliter):
    x_axis = dataframe.drop(columns=spliter, axis="columns")
    y_axis = dataframe[spliter]
    return x_axis, y_axis


# Splits the dataframe into train/test with ratio
def train_test_model(x_axis, y_axis, ratio):
    return train_test_split(x_axis, y_axis, test_size=1 - ratio, train_size=ratio)


# one hot encodes all categorical columns and leaves the numerical untouched
def one_hot_encoded(data_frame):
    return pd.get_dummies(data=data_frame, columns=get_categorical_columns(data_frame))


def column_transformer(data_frame):
    scaler = StandardScaler()
    for i in get_numerical_columns(data_frame=data_frame):
        data_frame[i] = scaler.fit_transform(data_frame[i])  # problem with random index from split train test
        # expected 2D array got a 1D array
    one_hot_encoded(data_frame=data_frame)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = load_data_set(dataset_name=DATASET_PATH, erase_columns=ERASE_COLUMNS)
    treated_df = treat_values(data_frame=df)
    x, y = split_data_frame(dataframe=treated_df, spliter="Fraud")
    x_train, x_test, y_train, y_test = train_test_model(x_axis=x, y_axis=y, ratio=0.9)
    column_transformer(data_frame=x_train)
    one_hot_encoded(data_frame=x_train)
    # TODO("Simple Imputer lib and Standard Scaler for numerical transformer")
