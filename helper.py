import numpy as np

def load_data(data_file, movies_file, skiprows = 0, m_cols = 1):
    '''
    Function loads movie and all ratings data from input
    files and returns data, genres, movies_data in numpy ndarrays.

    Inputs:
    data_file: filename for all ratings data
    movies_file: filename for movies
    skiprows: number of rows to skip in reading any input file
    m_cols: the column to start loading in movies file

    Outputs:
    data: the data from all ratings data file as numpy ndarray
    genres: returns a list of genres in the same order as labeled in movies_data
    movies_data: the data from each movie (removed movie id column)
    '''
    genres = np.loadtxt(movies_file, dtype = 'str', skiprows = skiprows,
    usecols = (m_cols,), delimiter = '\t', encoding="cp1252")
    movies_data = np.loadtxt(movies_file, skiprows = skiprows, usecols = m_cols + 1,
    delimiter = '\t', encoding="cp1252")

    data = np.loadtxt(data_file, skiprows = skiprows, delimiter = '\t')

    return data, genres, movies_data

# Example usage of load_data
# data, genres, movies_data = load_data('data/data.txt', 'data/movies.txt')


def error(y, y_pred):
  '''
  Returns classification error, given the actual and predicted y-values.
  '''
  return np.mean(y != y_pred)

def accuracy(y, y_pred):
  '''
  Returns classification accuracy, given the actual and predicted y-values.
  '''
  return np.mean(y == y_pred)
