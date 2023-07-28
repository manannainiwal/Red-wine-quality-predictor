# Data collection

I collected this data set from Kaggle website. The datasets is related to red variants of the Portuguese "Vinho Verde" wine. The data set have input variables as :

1 - fixed acidity

2 - volatile acidity

3 - citric acid

4 - residual sugar

5 - chlorides

6 - free sulfur dioxide

7 - total sulfur dioxide

8 - density

9 - pH

10 - sulphates

11 - alcohol

and Output variable as :

12 - quality

The data set have 1599 rows, 11 Features and 1 Target column.

link to the data base repository is https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009


# Exploratory data analysis and Feature Engineering


1) Data formatting : Firstly, I checked the data type of all columns to check that wheather data is in correct format or not. After finding that the data is in correct    format I take the basic mathematical description of the data using .describe() function.

2) Handling Missing Values : next step was checking and handling missing values. I found no missing data in the entire data set so we dont have to either drop or          impute and values in nan.

3) Handling Duplicates : then, I checked for presence of any duplicate data in the data set. I found 240 rows as duplicated data in the data set. I dropped the            duplicate data keeping only the first row of any duplicate data.

4) Handling Outliers : I have treat outliers with Inter quartile range. for this I have created two functions ie. outlier_finder that detect and print the outliers        present in the column and then we make a function named outlier_remover that removes those outliers. then we find and remove outliers from every column.

# Feature Selection

Firstly, I print the correlation matrix for our data frame to see the pearson correlation between our features. After this I plotted the Heatmap and decided to choose the cutoff pearson correlation value to be 6 for selecting features. Every features that have correlation higher then 6 among themselves we choose one of them based on their correlation with target column and drop the other. We dropped four features named 'pH', 'fixed acidity', 'citric acid', 'free sulfur dioxide'.

# Feature Scaling

After seeing the distribution of data we can conclude that Euclidean distance based models can perform well in this dataset so I choose to scale the data. I used MinMax Scaler to scale the data.
After scaling the data we make the target set into 0 and 1 that is good or bad wine by setting the rule that any quality score greater then 7 is considered as good wine else it will be a bad wine.

# Balancing the Dataset

I found that the dataset is imbalanced with bad wine count of 125 and good wine count of 860. So I use SMOTE technique to oversample the minority class and balance the dataset so that our model don't get biased.
After balancing the data set we divide the dataset into features(X) and target(y).

# Train Test Split

After dividing the data set into features and target set we do train test split. I keep 20% data for testing and remaining 80% data for training the model.

# Model Selection

I consider 5 Classifiers for model selection. Those classifiers are Random forest, XGBoost, K nearest neighbors, decision tree, SGDClassifier. I test them taking accuracy as my performance parameter. I found their scores as :

SGDClassifier()	                      0.845930
DecisionTreeClassifier()	      0.872093
XGBClassifier()	                      0.921512
KNeighborsClassifier(n_neighbors=1)   0.933140
RandomForestClassifier()	      0.944767

After this we choose top three performing models and check their accuracy using Cross validation with number of cross validation = 10 and found that XGBoost Classifier is performing the best with mean score of 92.9. So we choose our model as XGBoost Classifier.

# Hyperparameter Tuning

I consider four parameters to do hyperparameter tuning those are learning_rate, max_depth, min_child_weight, gamma. Firstly, I used Randomized search CV to find the range of values of parameters that are good for my model, then used Grid search CV to find the best possible combination of hyperparameters for my model.

# Checking Model Accuracy

For Checking the model's performance I consider three scores that are precision, recall and F1 score. I get precision of 0.9, Recall of 0.96 and F1 score of 0.933. finally, I check models score on test set and got the accuracy of 91.86. lastly I printed the Confusion matrix of the model:

[611  74]
[ 22 669]
