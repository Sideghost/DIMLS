import pandas as pd
from sklearn.model_selection import train_test_split

DATASET = "dataSet/warranty_claims.csv"


# Loads a CSV into a data frame.
# Erases the unnecessary columns as in 'ID'.
def load_data_set(datasetname):
    return pd.read_csv(datasetname).drop(columns="ID")


# Splits the data frame into X input and Y output
def split_data_frame(dataframe):
    x_axis = dataframe.drop("Fraud", axis="columns")
    y_axis = dataframe["Fraud"]
    return x_axis, y_axis


# Splits the dataframe into train/test with ratio
def train_test_model(x, y, ratio):
    return train_test_split(x, y, train_size=ratio, test_size=1-ratio)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = load_data_set(DATASET)
    df.fillna(df.median())
    input, output = split_data_frame(df)
    print(train_test_model(input, output, 0.9))
