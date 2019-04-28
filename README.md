# Computing Classification Evaluation Metrics in R
by Said Bleik, Shaheen Gauher, Data Scientists at Microsoft
https://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html

Evaluation metrics are the key to understanding how your classification model performs when applied to a test dataset. In what follows, we present a tutorial on how to compute common metrics that are often used in evaluation, in addition to metrics generated from random classifiers, which help in justifying the value added by your predictive model, especially in cases where the common metrics suggest otherwise.

Creating the Confusion Matrix
Accuracy
Per-class Precision, Recall, and F-1
Macro-averaged Metrics
One-vs-all Matrices
Average Accuracy
Micro-averaged Metrics
Evaluation on Highly Imbalanced Datasets
Majority-class Metrics
Random-guess Metrics
Kappa Statistic
Custom R Evaluation Module in Azure Machine Learning

## Creating the Confusion Matrix
We will start by creating a confusion matrix from simulated classification results. The confusion matrix provides a tabular summary of the actual class labels vs. the predicted ones. The test set we are evaluating on contains 100 instances which are assigned to one of 3 classes 
a, b or c.
```r
set.seed(0)
 actual = c('a','b','c')[runif(100, 1,4)] # actual labels
 predicted = actual # predicted labels
 predicted[runif(30,1,100)] = actual[runif(30,1,100)]  # introduce incorrect predictions
 cm = as.matrix(table(Actual = actual, Predicted = predicted)) # create the confusion matrix
 cm
##       Predicted
## Actual  a  b  c
##      a 24  2  1
##      b  3 30  4
##      c  0  5 31
```

Next we will define some basic variables that will be needed to compute the evaluation metrics.
```r
  n = sum(cm) # number of instances
 nc = nrow(cm) # number of classes
 diag = diag(cm) # number of correctly classified instances per class 
 rowsums = apply(cm, 1, sum) # number of instances per class
 colsums = apply(cm, 2, sum) # number of predictions per class
 p = rowsums / n # distribution of instances over the actual classes
 q = colsums / n # distribution of instances over the predicted classes
```
## Accuracy
A key metric to start with is the overall classification accuracy. It is defined as the fraction of instances that are correctly classified.
```
 accuracy = sum(diag) / n 
 accuracy 
## [1] 0.85
```

## Per-class Precision, Recall, and F-1
In order to assess the performance with respect to every class in the dataset, we will compute common per-class metrics such as precision, recall, and the F-1 score. These metrics are particularly useful when the class labels are not uniformly distributed (most instances belong to one class, for example). In such cases, accuracy could be misleading as one could predict the dominant class most of the time and still achieve a relatively high overall accuracy but very low precision or recall for other classes. Precision is defined as the fraction of correct predictions for a certain class, whereas recall is the fraction of instances of a class that were correctly predicted. Notice that there is an obvious trade off between these 2 metrics. When a classifier attempts to predict one class, say class a, most of the time, it will achieve a high recall for a (most of the instances of that class will be identified). However, instances of other classes will most likely be incorrectly predicted as a in that process, resulting in a lower precision for a. In addition to precision and recall, the F-1 score is also commonly reported. It is defined as the harmonic mean (or a weighted average) of precision and recall.
```r
 precision = diag / colsums 
 recall = diag / rowsums 
 f1 = 2 * precision * recall / (precision + recall) 
 data.frame(precision, recall, f1) 
##   precision    recall        f1
## a 0.8888889 0.8888889 0.8888889
## b 0.8108108 0.8108108 0.8108108
## c 0.8611111 0.8611111 0.8611111
```
Note that this is an example of multi-class classification evaluation and that some of the variables we compute are vectors that contain multiple values representing each class. For example, precision contains 3 values corresponding to the classes a, b, and c. The code can generalize to any number of classes. However, in binary classification tasks, one would look at the values of the positive class when reporting such metrics. In that case, the overall precision, recall and F-1, are those of the positive class.

## Macro-averaged Metrics
The per-class metrics can be averaged over all the classes resulting in macro-averaged precision, recall and F-1.
```r
  macroPrecision = mean(precision)
  macroRecall = mean(recall)
  macroF1 = mean(f1)
  data.frame(macroPrecision, macroRecall, macroF1)
##   macroPrecision macroRecall   macroF1
## 1      0.8536036   0.8536036 0.8536036
```
## One-vs-all
When the instances are not uniformly distributed over the classes, it is useful to look at the performance of the classifier with respect to one class at a time before averaging the metrics. In the following script, we will compute the one-vs-all confusion matrix for each class (3 matrices in this case). You can think of the problem as 3 binary classification tasks where one class is considered the positive class while the combination of all the other classes make up the negative class.
```r
  oneVsAll = lapply(1 : nc,
                      function(i){
                        v = c(cm[i,i],
                              rowsums[i] - cm[i,i],
                              colsums[i] - cm[i,i],
                              n-rowsums[i] - colsums[i] + cm[i,i]);
                        return(matrix(v, nrow = 2, byrow = T))})
 oneVsAll
## [[1]]
##      [,1] [,2]
## [1,]   24    3
## [2,]    3   70
## 
## [[2]]
##      [,1] [,2]
## [1,]   30    7
## [2,]    7   56
## 
## [[3]]
##      [,1] [,2]
## [1,]   31    5
## [2,]    5   59
```
Summing up the values of these 3 matrices results in one confusion matrix and allows us to compute weighted metrics such as average accuracy and micro-averaged metrics.
```r
 s = matrix(0, nrow = 2, ncol = 2)
 for(i in 1 : nc){s = s + oneVsAll[[i]]}
 s
##      [,1] [,2]
## [1,]   85   15
## [2,]   15  185
```
## Average Accuracy
Similar to the overall accuracy, the average accuracy is defined as the fraction of correctly classified instances in the sum of one-vs-all matrices matrix.
```r
 avgAccuracy = sum(diag(s)) / sum(s)
 avgAccuracy
## [1] 0.9
```

## Micro-averaged Metrics
The micro-averaged precision, recall, and F-1 can also be computed from the matrix above. Compared to unweighted macro-averaging, micro-averaging favors classes with a larger number of instances. Because the sum of the one-vs-all matrices is a symmetric matrix, the micro-averaged precision, recall, and F-1 wil be the same.

```r
 micro_prf = (diag(s) / apply(s,1, sum))[1];
 micro_prf
## [1] 0.85
```

## Evaluation on Highly Imbalanced Datasets
Many times, your common evaluation metrics suggest a model is performing poorly. Nevertheless, you believe that the predictions can potentially add considerable value to your business or research work. This is usually the case in scenarios where the data is not equally representative of all classes, such as rare event classification, or classification of highly imbalanced datasets. In such cases, a model might be biased towards the majority class, while the performance relative to the less occurring class labels is seemingly unacceptable. One way to justify the results of such classifiers is by comparing them to those of baseline classifiers and showing that they are indeed better than random chance predictions.

## Majority-class Metrics
When a class dominates a dataset, predicting the majority class for all instances in the test set ensures a high overall accuracy as most of the labels will be predicted correctly. If having a high accuracy is your sole objective, then a naive majority-class model can be better than a learned model in many cases. However, this will not be very useful in practice, as it is often the case that you are more interested in making correct predictions for the other classes (predicting an imminent failure in a device, for example). Below we calculate the expected results of a majortiy-class classifier applied on the same dataset. The overall accuracy of this classifier, also called No Information Rate (NIR), and its precision on the majority class are equal to the proportion of instances that belong to the majority class. Recall on the majority class is equal to 1 (all majority class instances will be predicted correctly).
```r
 mcIndex = which(rowsums==max(rowsums))[1] # majority-class index
 mcAccuracy = as.numeric(p[mcIndex]) 
 mcRecall = 0*p;  mcRecall[mcIndex] = 1
 mcPrecision = 0*p; mcPrecision[mcIndex] = p[mcIndex]
 mcF1 = 0*p; mcF1[mcIndex] = 2 * mcPrecision[mcIndex] / (mcPrecision[mcIndex] + 1)
 mcIndex
## b 
## 2
 mcAccuracy
## [1] 0.37
 data.frame(mcRecall, mcPrecision, mcF1) 
##   mcRecall mcPrecision     mcF1
## a        0        0.00 0.000000
## b        1        0.37 0.540146
## c        0        0.00 0.000000
```

## Random-guess Metrics
Another baseline classifier is one that predicts labels randomly (no learning involved). We will call this a random-guess classifier. It is also useful to compare your model to, for the same reasons discussed above. If you were to make a random guess and predict any of the possible labels, the expected overall accuracy and recall for all classes would be the same as the probability of picking a certain class. The expected precision would be the same as the probability that a chosen label is actually correct, which is equal to the proportion of instances that belong to a class. For example, given 
n
c
 classes, you predict 
n
n
c
 instances as class a instances and expect them to be correctly classified with probability 
p
a
.

To help illustrate this, we can create the expected confusion matrix:
```r
 (n / nc) * matrix(rep(p, nc), nc, nc, byrow=F)
##          [,1]     [,2]     [,3]
## [1,]  9.00000  9.00000  9.00000
## [2,] 12.33333 12.33333 12.33333
## [3,] 12.00000 12.00000 12.00000
```
Using some algebra, we can verify that the metrics can be computed as follows:
```r
  rgAccuracy = 1 / nc
  rgPrecision = p
  rgRecall = 0*p + 1 / nc
  rgF1 = 2 * p / (nc * p + 1)
 rgAccuracy
## [1] 0.3333333
 data.frame(rgPrecision, rgRecall, rgF1)
##   rgPrecision  rgRecall      rgF1
## a        0.27 0.3333333 0.2983425
## b        0.37 0.3333333 0.3507109
## c        0.36 0.3333333 0.3461538
```
Sometimes you know the prior distribution of the data and would like to use that information when making a random-guess prediction. This would help in having a more reliable baseline to compare to, especially when the data distribution is skewed (imbalanced classes). We will call this baseline a random-weighted-guess classifier. In the following, we will assume that the prior distribution of the data is the same as that of the test set (p). In other cases, you might want to use the distribution of the training set, or any other given class proportions you believe are appropriate. We will therefore make predictions based on those proportions, that is predict a certain label according to its probability of occurrence in the data. For example, the number of class a instances that would be predicted as class a instances is 

```r
 n * p %*% t(p)
##         a     b     c
## [1,] 7.29  9.99  9.72
## [2,] 9.99 13.69 13.32
## [3,] 9.72 13.32 12.96
```
The results can easily be generalized using some basic algebra, from which we can conclude that the expected accuracy is equal to the sum of squares of the class proportions p, while precision and recall are equal to p.
```r
 rwgAccurcy = sum(p^2)
 rwgPrecision = p
 rwgRecall = p
 rwgF1 = p
 rwgAccurcy
## [1] 0.3394
 data.frame(rwgPrecision, rwgRecall, rwgF1)
##   rwgPrecision rwgRecall rwgF1
## a         0.27      0.27  0.27
## b         0.37      0.37  0.37
## c         0.36      0.36  0.36
```
## Kappa Statistic
Similarly, we can compute the Kappa statistic, which is a measure of agreement between the predictions and the actual labels. It can also be interpreted as a comparison of the overall acurracy to the expected random chance accuracy. The higher the Kappa metric is, the better your classifier is compared to a random chance classifier. The intuition behind the Kappa statistic is the same as the random guess metrics we have just discussed. However, the expected accuracy used in computing Kappa is based on both the actual and predicted distributions. That is, we predict 
Kappa is defined as the difference between the overall accuracy and the expected accuracy divided by 1 minus the expected accuracy.
```r
 expAccuracy = sum(p*q)
 kappa = (accuracy - expAccuracy) / (1 - expAccuracy)
  kappa
## [1] 0.7729337
```
In general, you can can compare to any random baseline classifier by replacing the values of p and q with whatever distributions you think are fit for the comparison. By doing so, you are computing evaluation metrics based on your expectation of both the classifier and the actual distribution of the data.

Custom R Evaluation Module in Azure Machine Learning
We have created an Azure Machine Learning (AML) custom R evaluation module that can be imported and used in AML experiments. The module computes all metrics discussed in this article. You can find it in the Cortana Analytics Gallery. To use it, open and run the experiment in the AML studio. A module named Evaluate Model will show up in the Custom section.
