import numpy as np

def load_data(data_file, train_file, test_file, movies_file, skiprows = 0,
m_cols = 2):
    '''
    Function loads training and test data and movie and info data from input
    files and returns x_train, y_train, and x_test in numpy ndarrays.

    Inputs:
    data_file: filename for all ratings data
    movies_file: filename for movies
    train_file: filename for training ratings data
    test_file: filename for testing ratings data
    skiprows: number of rows to skip in reading any input file
    m_cols: the column to start loading in movies file

    Outputs:
    data: the data from all ratings data file as numpy ndarray
    X_train: x values for training set as numpy ndarray
    y_train: labels for x values in training set as numpy ndarray
    X_test: x values for testing set as numpy ndarray
    y_test: x values for testing set as numpy ndarray
    '''
    movies_data = np.loadtxt(movies_file, skiprows = skiprows, usecols = m_cols,
     delimiter = '\t', encoding="cp1252")
    data = np.loadtxt(data_file, skiprows = skiprows, delimiter = '\t')
    train_data = np.loadtxt(train_file, skiprows = skiprows, delimiter = '\t')
    test_data = np.loadtxt(test_file, skiprows = skiprows, delimiter = '\t')

    X_train = train_data[:, :-1]
    y_train = train_data[:,-1]

    X_test = test_data[:, :-1]
    y_test = test_data[:,-1]

    return data, X_train, y_train, X_test, y_test

# Example usage of load_data
'''
data, X_train, y_train, X_test, y_test = load_data('data/data.txt',
'data/train.txt', 'data/test.txt', 'data/movies.txt')
'''
