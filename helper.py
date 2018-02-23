import numpy as np

def load_data(data_file, movies_file, skiprows = 0):
    '''
    Function loads movie and all ratings data from input
    files and returns data, genres, movies_data in numpy ndarrays.

    Inputs:
    data_file: filename for all ratings data
    movies_file: filename for movies
    skiprows: number of rows to skip in reading any input file

    Outputs:
    data: the data from all ratings data file as numpy ndarray
    titles: returns a list of genres in the same order as labeled in movies_data
    movies_data: the data from each movie (removed movie id column)
    genres: returns a list of the 19 genres in order that movies_data is labeled
    '''
    movies = np.loadtxt(movies_file, dtype = 'str', skiprows = skiprows, delimiter = '\t', encoding="cp1252")
    titles = movies[:, 1]
    movies_data = movies[:, 2:].astype(float)
    genres = ["Unknown", "Action", "Adventure", "Animation", "Childrens",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
    "War", "Western"]

    data = np.loadtxt(data_file, skiprows = skiprows, delimiter = '\t')

    return data, titles, movies_data, genres

# Example usage of load_data
data, titles, movies_data, genres = load_data('data/data.txt', 'data/movies.txt')

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
