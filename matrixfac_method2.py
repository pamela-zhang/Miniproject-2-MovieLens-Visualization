# Second method of matrix factorization.
# (basically just the first method with bias/offset terms)

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data, get_popular_best_movies
from sklearn.preprocessing import scale

def grad_U(Ui, Yij, Vj, ai, bj, u, reg, eta):
  """
  Takes as input Ui (the ith row of U), a training point Yij, the column vector
  Vj (jth column of V^T), reg (the regularization parameter lambda), and eta
  (the learning rate).

  Returns the gradient of the regularized loss function with respect to Ui
  multiplied by eta.
  """
  return eta * (reg * Ui - ((Yij - u) - (np.dot(Ui, Vj) + ai + bj)) * Vj)

def get_bias_terms(M, N, Y):
  """
  M: number of users
  N: number of movies
  Y: training data matrix Y containing rows (i, j, Y_ij) where Y_ij is user
  i's rating on movie j

  Returns u (global bias), a (user-specific deviations), b (movie-specific deviations)
  """

  # Records user ratings in order of user (total rating score, number of ratings)
  user_ratings = np.zeros((M, 2))
  movie_ratings = np.zeros((N, 2))
  for d in Y:
    user_ratings[(int(d[0])-1)][0] += d[2]
    user_ratings[(int(d[0])-1)][1] += 1

    movie_ratings[int(d[1] - 1)][0] += d[2]
    movie_ratings[int(d[1] - 1)][1] += 1

  # Find average ratings for each user and then for each movie
  user_avgs = np.zeros(M)
  movie_avgs = np.zeros(N)
  for i, m in enumerate(user_ratings):
    if m[1] != 0:
      user_avgs[i] = m[0] / m[1]
  for i, m in enumerate(movie_ratings):
    if m[1] != 0:
      movie_avgs[i] = m[0] / m[1]

  # Bias vector represents user-specific bias compared to global bias
  u = np.mean(user_avgs)
  a = user_avgs - u
  b = movie_avgs - u
  return u, a, b

def grad_V(Vj, Yij, Ui, ai, bj, u, reg, eta):
  """
  Takes as input the column vector Vj (jth column of V^T), a training point Yij,
  Ui (the ith row of U), reg (the regularization parameter lambda), and eta (the
  learning rate).

  Returns the gradient of the regularized loss function with respect to Vj
  multiplied by eta.
  """
  return eta * (reg * Vj - ((Yij - u) - (np.dot(Ui, Vj) + ai + bj)) * Ui)

def get_err(U, V, Y, u, a, b, reg=0.0):
  """
  Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a
  user, j is the index of a movie, and Y_ij is user i's rating of movie j and
  user/movie matrices U and V.

  Returns the mean regularized squared-error of predictions made by estimating
  Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
  """
  t_1 = reg * (np.linalg.norm(U, 'fro') ** 2 + np.linalg.norm(V, 'fro') ** 2 +
    np.linalg.norm(a) ** 2 + np.linalg.norm(b) ** 2)
  t_2 = 0
  for y in Y:
    i, j, Yij = y[0], y[1], y[2]
    t_2 += ((Yij-u) - (np.dot(U[:, i-1], V[:, j-1]) + a[i-1] + b[j-1])) ** 2
  return 0.5 * (t_1 + t_2) / Y.shape[0]

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
  """
  Given a training data matrix Y containing rows (i, j, Y_ij) where Y_ij is user
  i's rating on movie j, learns an M x K matrix U and N x K matrix V such that
  rating Y_ij is approximated by (UV^T)_ij.

  Uses a learning rate of <eta> and regularization of <reg>. Stops after
  <max_epochs> epochs, or once the magnitude of the decrease in regularized MSE
  between epochs is smaller than a fraction <eps> of the decrease in MSE after
  the first epoch.

  Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE of
  the model.
  """

  # Find bias vectors
  u, a, b = get_bias_terms(M, N, Y)

  # Initialize U and V to be small random numbers.
  U = np.random.uniform(-0.5, 0.5, (K, M))
  V = np.random.uniform(-0.5, 0.5, (K, N))

  loss = get_err(U, V, Y, u, a, b, reg=reg)
  init_loss_reduction = 0

  for t in range(1, max_epochs + 1):
    # Randomly shuffle the training data.
    order = np.random.permutation(Y.shape[0])

    # Perform SGD.
    for z in order:
      i, j, Yij = Y[z, 0], Y[z, 1], Y[z, 2]
      U[:, i-1] -= grad_U(U[:, i-1], Yij, V[:, j-1], u, a[i-1], b[j-1], reg, eta)
      V[:, j-1] -= grad_V(V[:, j-1], Yij, U[:, i-1], u, a[i-1], b[j-1], reg, eta)

    # Check for early stopping condition.
    new_loss = get_err(U, V, Y, u, a, b, reg=reg)
    if t == 1:
      init_loss_reduction = loss - new_loss
    elif (loss - new_loss) / init_loss_reduction <= eps:
      break
    loss = new_loss

  return U, V, get_err(U, V, Y, u, a, b)


def main():
  # Load data from txt files.
  data, titles, movies_data, genres = load_data('data/data.txt',
    'data/movies.txt')
  Y_train = np.loadtxt('data/train.txt').astype(int)
  Y_test = np.loadtxt('data/test.txt').astype(int)

  # ----------------------------------------------
  # STEP 1. LEARN U AND V
  # ----------------------------------------------

  M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int)  # users
  N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int)  # movies
  print("Factorizing with ", M, " users, ", N, " movies.")

  K = 20
  reg = 0.1  # regularization parameter (lambda)
  eta = 0.01  # learning rate
  E_in = []
  E_out = []

  # Train model.
  U, V, err = train_model(M, N, K, eta, reg, Y_train)
  E_in.append(err)
  u, a, b = get_bias_terms(M, N, Y_test)
  E_out.append(get_err(U, V, Y_test, u, a, b))
  print('Errors (Method 2): ', E_in, E_out)

  '''
  # Plot to determine optimal parameters.
  plt.plot(regs, E_in, 'o-', label='$E_{in}$')
  plt.plot(regs, E_out, 'o-', label='$E_{out}$')
  plt.title('Error vs. Regularization')
  plt.xlabel('Regularization Parameter (lambda)')
  plt.ylabel('Error')
  plt.legend()
  plt.savefig('images/method2_optimize_regularization.png', dpi=300)
  '''

  # ----------------------------------------------
  # STEP 2. PROJECT U AND V DOWN TO 2 DIMENSIONS
  # ----------------------------------------------

  # Mean center V and apply the same shifts to U.
  for i in range(len(V)):
    V[i, :] -= np.mean(V[i, :])
    U[i, :] -= np.mean(V[i, :])

  # Get singular value decomposition of V. (V = A s B)
  A, s, B = np.linalg.svd(V, full_matrices=False)

  # Project U and V onto the best 2 dimensions.
  V_proj = np.dot(A[:, :2].T, V)
  U_proj = np.dot(A[:, :2].T, U)

  # Rescale dimensions so that each row of U_proj has unit variance.
  U_proj = scale(U_proj, axis=1, with_mean=False)

  # ----------------------------------------------
  # STEP 3. PLOT PROJECTED U AND V
  # ----------------------------------------------

  # Plot 10 movies of our choice.
  movies = [0, 49, 68, 70, 94, 97, 126, 248, 312, 385]
  for m in movies:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Movies (Method 2)')
  plt.savefig('images/method2_vis_our10movies.png', bbox_inches='tight', dpi=300)

  # Get the indices (movie ID - 1) of the 10 most popular and 10 best movies.
  most_pop_movies, best_movies = get_popular_best_movies(data, movies_data)

  # Plot the 10 most popular movies.
  plt.figure()
  for m in most_pop_movies:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Most Popular Movies (Method 2)')
  plt.savefig('images/method2_vis_10mostpopular.png', bbox_inches='tight', dpi=300)

  # Plot the 10 best movies.
  plt.figure()
  for m in best_movies:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Best Movies (Method 2)')
  plt.savefig('images/method2_vis_10best.png', bbox_inches='tight', dpi=300)

  # Plot 10 Western movies.
  plt.figure()
  for m in (np.where([movies_data[:, 18] == 1])[1])[:10]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Western Movies (Method 2)')
  plt.savefig('images/method2_vis_western.png', bbox_inches='tight', dpi=300)

  # Plot 10 Animation movies.
  plt.figure()
  for m in (np.where([movies_data[:, 3] == 1])[1])[:10]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Animation Movies (Method 2)')
  plt.savefig('images/method2_vis_animation.png', bbox_inches='tight', dpi=300)

  # Plot 10 Film-Noir movies.
  plt.figure()
  for m in (np.where([movies_data[:, 10] == 1])[1])[:10]:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Film-Noir Movies (Method 2)')
  plt.savefig('images/method2_vis_filmnoir.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()
