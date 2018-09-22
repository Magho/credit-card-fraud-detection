## Project Description 
This notebook uses Credit card transsition data to determine if the transition is valid or not

## packages used 
- sys
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- sklearn

## data set used 
[Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Algorithms used
### Unsupervised Outlier Detection :

Now that we have processed our data, we can begin deploying our machine learning algorithms. We will use the following techniques:

- #### Local Outlier Factor (LOF)

    The anomaly score of each sample is called Local Outlier Factor. It measures the local deviation of density of a given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood.

- #### Isolation Forest Algorithm

    The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.
    
## Algorithms tested on 0.1, 0.5, 0.75 , 1.0  of the data and tested on a sample of the data with number of valid transition equal 100 * times fraud transisions size (this was the best accuracy of all)