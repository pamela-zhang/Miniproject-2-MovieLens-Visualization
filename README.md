# MovieLens-Visualization
## CS 155 MiniProject 2
### Description of Miniproject
In this project, you will be creating visualizations of the MovieLens data set
using matrix factorization. The MovieLens data set consists of 100,000 ratings
from 943 users on 1682 movies, where each user has rated at least 20 movies.
The last 19 fields are various movie genres. Here, a 1 indicates the movie is
of the given genre, while a 0 indicates that it is not. Note that movies can be
in several genres at once. The movie ids correspond to the movie ids specified
in the data.txt file and range from 1 to 1682.

### Project Tasks
First we created some basic visualizations of the MovieLens dataset, including
emphasis on the most popular or best movies. Next, we use matrix factorization
to learn the model, and optimize it by adding the bias terms and trying
"off the shelf" techniques related to collaborative filtering.

### Packages
Need NumPy version at least 1.14.0
