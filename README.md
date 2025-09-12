# progat-xg

Progressive Graph Attention Networks for Interpretable Expected Goals Analysis in Soccer


### Dataset
The dataset is built from the Statsbomb dataset. The idea is to get all the ball possessions from all the teams from all the games in the dataset.
The full dataset contains:

Total Possessions: 1623349
Total Shots: 55108

As you can see the dataset is very very unmbalanced. For this reason I will try different approaches to improve the performance of the model.
So I will try both to use the whole dataset with the obvious problem of overfitting and with random sampling to balance the dataset but loosing the statistical learning informations.

### Alternative Attention

Try to implement an alternative attention formula adapted to this task.

xz