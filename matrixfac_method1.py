# First method of matrix factorization.
# (basically just the method from HW 5)

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data, get_popular_best_movies
from sklearn.preprocessing import scale

def grad_U(Ui, Yij, Vj, reg, eta):
  """
  Takes as input Ui (the ith row of U), a training point Yij, the column vector
  Vj (jth column of V^T), reg (the regularization parameter lambda), and eta
  (the learning rate).

  Returns the gradient of the regularized loss function with respect to Ui
  multiplied by eta.
  """
  return eta * (reg * Ui - (Yij - np.dot(Ui, Vj)) * Vj)

def grad_V(Vj, Yij, Ui, reg, eta):
  """
  Takes as input the column vector Vj (jth column of V^T), a training point Yij,
  Ui (the ith row of U), reg (the regularization parameter lambda), and eta (the
  learning rate).

  Returns the gradient of the regularized loss function with respect to Vj
  multiplied by eta.
  """
  return eta * (reg * Vj - (Yij - np.dot(Ui, Vj)) * Ui)

def get_err(U, V, Y, reg=0.0):
  """
  Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a
  user, j is the index of a movie, and Y_ij is user i's rating of movie j and
  user/movie matrices U and V.

  Returns the mean regularized squared-error of predictions made by estimating
  Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
  """
  a = reg * (np.linalg.norm(U, 'fro') ** 2 + np.linalg.norm(V, 'fro') ** 2)
  b = 0
  for y in Y:
    i, j, Yij = y[0], y[1], y[2]
    b += (Yij - np.dot(U[:, i-1], V[:, j-1])) ** 2
  return 0.5 * (a + b) / Y.shape[0]

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
  # Initialize U and V to be small random numbers.
  U = np.random.uniform(-0.5, 0.5, (K, M))
  V = np.random.uniform(-0.5, 0.5, (K, N))

  loss = get_err(U, V, Y, reg=reg)
  init_loss_reduction = 0

  for t in range(1, max_epochs + 1):
    # Randomly shuffle the training data.
    order = np.random.permutation(Y.shape[0])

    # Perform SGD.
    for a in order:
      i, j, Yij = Y[a, 0], Y[a, 1], Y[a, 2]
      U[:, i-1] -= grad_U(U[:, i-1], Yij, V[:, j-1], reg, eta)
      V[:, j-1] -= grad_V(V[:, j-1], Yij, U[:, i-1], reg, eta)

    # Check for early stopping condition.
    new_loss = get_err(U, V, Y, reg=reg)
    if t == 1:
      init_loss_reduction = loss - new_loss
    elif (loss - new_loss) / init_loss_reduction <= eps:
      break
    loss = new_loss

  return U, V, get_err(U, V, Y)


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
  E_out.append(get_err(U, V, Y_test))

  '''
  # Plot to determine optimal parameters.
  plt.plot(regs, E_in, 'o-', label='$E_{in}$')
  plt.plot(regs, E_out, 'o-', label='$E_{out}$')
  plt.title('Error vs. Regularization')
  plt.xlabel('Regularization Parameter (lambda)')
  plt.ylabel('Error')
  plt.legend()
  plt.savefig('images/method1_optimize_regularization.png', dpi=300)
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
  plt.title('Visualization of 10 Movies (HW5 Matrix Factorization)')
  plt.savefig('images/method1_vis_our10movies.png', bbox_inches='tight', dpi=300)

  # Get the indices (movie ID - 1) of the 10 most popular and 10 best movies.
  most_pop_movies, best_movies = get_popular_best_movies(data, movies_data)

  # Plot the 10 most popular movies.
  plt.figure()
  for m in most_pop_movies:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Most Popular Movies (HW5 Matrix Factorization)')
  plt.savefig('images/method1_vis_10mostpopular.png', bbox_inches='tight', dpi=300)

  # Plot the 10 best movies.
  plt.figure()
  for m in best_movies:
    plt.plot(V_proj[0, m], V_proj[1, m], 'o', color='#ffa500')
    plt.annotate(titles[m].strip('"'), (V_proj[0, m], V_proj[1, m]))
  plt.title('Visualization of 10 Best Movies (HW5 Matrix Factorization)')
  plt.savefig('images/method1_vis_10best.png', bbox_inches='tight', dpi=300)

  # TODO: Plot 10 movies from <GENRE 1>.

  # TODO: Plot 10 movies from <GENRE 2>.

  # TODO: Plot 10 movies from <GENRE 3>.


if __name__ == "__main__":
    main()
