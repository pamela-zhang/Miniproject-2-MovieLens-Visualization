# Third method for matrix factorization:
# off-the-shelf implementation using Surprise.

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data, get_popular_best_movies
from surprise import SVD, Dataset, Reader, accuracy

def main():

  # Load the movielens-100k dataset.
  data, titles, movies_data, genres = load_data('data/data.txt', 
    'data/movies.txt')
  reader = Reader(line_format='user item rating', sep='\t')
  dataset = Dataset.load_from_file('data/data.txt', reader=reader)
  trainset = dataset.build_full_trainset()

  test_data = Dataset.load_from_file('data/test.txt', reader=reader)                                
  testset = test_data.construct_testset(raw_testset=test_data.raw_ratings)    

  # ----------------------------------------------
  # LEARN U AND V & PROJECT TO 2D
  # ----------------------------------------------

  # Use SVD algorithm.
  algo = SVD(n_factors=2, n_epochs=300)

  algo.fit(trainset)
  test_pred = algo.test(testset)

  accuracy.rmse(test_pred)

  V_proj = algo.qi.T

  # ----------------------------------------------
  # PLOT PROJECTED U AND V
  # ----------------------------------------------

  # Plot 10 movies of our choice.
  for m in [0, 49, 68, 70, 94, 97, 126, 248, 312, 385]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Movies (Surprise)')
  plt.savefig('images/method3_vis_our10movies.png', bbox_inches='tight', dpi=300)

  # Get the indices (movie ID - 1) of the 10 most popular and 10 best movies.
  most_pop_movies, best_movies = get_popular_best_movies(data, movies_data)

  # Plot the 10 most popular movies.
  plt.figure()
  for m in most_pop_movies:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Most Popular Movies (Surprise)')
  plt.savefig('images/method3_vis_10mostpopular.png', bbox_inches='tight', dpi=300)

  # Plot the 10 best movies.
  plt.figure()
  for m in best_movies:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Best Movies (Surprise)')
  plt.savefig('images/method3_vis_10best.png', bbox_inches='tight', dpi=300)

  # Plot 10 Western movies.
  plt.figure()
  for m in (np.where([movies_data[:, 18] == 1])[1])[:10]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Western Movies (Surprise)')
  plt.savefig('images/method3_vis_western.png', bbox_inches='tight', dpi=300)

  # Plot 10 Animation movies.
  plt.figure()
  for m in (np.where([movies_data[:, 3] == 1])[1])[:10]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Animation Movies (Surprise)')
  plt.savefig('images/method3_vis_animation.png', bbox_inches='tight', dpi=300)

  # Plot 10 Film-Noir movies.
  plt.figure()
  for m in (np.where([movies_data[:, 10] == 1])[1])[:10]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Film-Noir Movies (Surprise)')
  plt.savefig('images/method3_vis_filmnoir.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()
