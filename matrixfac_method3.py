# Third method for matrix factorization:
# off-the-shelf implementation using sklearn.decomposition.

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data, get_popular_best_movies
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


def main():

  # Load data from txt files.
  data, titles, movies_data, genres = load_data('data/data.txt', 'data/movies.txt')
  Y_train = np.loadtxt('data/train.txt').astype(int)
  Y_test = np.loadtxt('data/test.txt').astype(int)

  # ----------------------------------------------
  # LEARN U AND V AND PROJECT TO 2D
  # ----------------------------------------------

  pca = PCA(n_components=2)
  V_proj = pca.fit_transform(movies_data)
  V_proj = V_proj.T

  # ----------------------------------------------
  # PLOT PROJECTED U AND V
  # ----------------------------------------------

  # Plot 10 movies of our choice.
  for m in [0, 49, 68, 70, 94, 97, 126, 248, 312, 385]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Movies (sklearn PCA)')
  plt.savefig('images/method3_vis_our10movies.png', bbox_inches='tight', dpi=300)

  # Get the indices (movie ID - 1) of the 10 most popular and 10 best movies.
  most_pop_movies, best_movies = get_popular_best_movies(data, movies_data)

  # Plot the 10 most popular movies.
  plt.figure()
  for m in most_pop_movies:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Most Popular Movies (sklearn PCA)')
  plt.savefig('images/method3_vis_10mostpopular.png', bbox_inches='tight', dpi=300)

  # Plot the 10 best movies.
  plt.figure()
  for m in best_movies:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Best Movies (sklearn PCA)')
  plt.savefig('images/method3_vis_10best.png', bbox_inches='tight', dpi=300)

  # Plot 10 Western movies.
  plt.figure()
  for m in (np.where([movies_data[:, 18] == 1])[1])[:10]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Western Movies (sklearn PCA)')
  plt.savefig('images/method3_vis_western.png', bbox_inches='tight', dpi=300)

  # Plot 10 Animation movies.
  plt.figure()
  for m in (np.where([movies_data[:, 3] == 1])[1])[:10]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Animation Movies (sklearn PCA)')
  plt.savefig('images/method3_vis_animation.png', bbox_inches='tight', dpi=300)

  # Plot 10 Film-Noir movies.
  plt.figure()
  for m in (np.where([movies_data[:, 10] == 1])[1])[:10]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Film-Noir Movies (sklearn PCA)')
  plt.savefig('images/method3_vis_filmnoir.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()
