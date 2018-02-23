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
# data, titles, movies_data, genres = load_data('data/data.txt', 'data/movies.txt')

def get_popular_best_movies(data, movies_data):
  # Array stores total of user ratings and number of ratings
  movie_ratings = np.zeros((len(movies_data), 2))

  # Loop through data set
  for d in data:
      # Add user's rating for corresponding movie
      movie_ratings[int(d[1] - 1)][0] += d[2]
      # Count of ratings per movie
      movie_ratings[int(d[1] - 1)][1] += 1

  '''
  Given an input array of movie ratings and scalar n_max, return the indices in 
  lst of the n_max highest values in the list.

  Input:
      - lst: list of values to sort and select from
      - n_max: desired n number of movies with certain attribute

  Output:
      - max_positions: indices of n_max movies with most number of ratings

  '''
  def n_max_pos(lst, n_max):
    sorted_pos = np.argsort(lst)
    max_positions = sorted_pos[-1*n_max:]
    return max_positions

  # Calculate average rating score for each movie
  avgs = np.zeros(len(movies_data))
  for i, m in enumerate(movie_ratings):
      # all_ratings[i] = m[1]
      if m[1] != 0:
          avgs[i] = m[0] / m[1]
        
  # Finding top 10 movies with the most ratings
  rating_per_mov = movie_ratings[:,1]
  most_ratings_pos = n_max_pos(rating_per_mov, 10)

  # Find average ratings of the 10 movies with most number of ratings
  most_ratings = avgs[most_ratings_pos]

  # Finding top 10 movies with highest average ratings
  best_movies_pos = n_max_pos(avgs, 10)
  best_avg = avgs[best_movies_pos]

  return most_ratings_pos, best_movies_pos
