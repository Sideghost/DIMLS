# Getter for all categorical columns
def get_categorical_columns(data_frame):
    return list(data_frame.select_dtypes(include=['object']).columns)


# Getter for all numerical columns
def get_numerical_columns(data_frame):
    return list(data_frame.select_dtypes(include=['float64']).columns)
