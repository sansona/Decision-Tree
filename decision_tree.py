"""
From "scratch" implementation of decision tree classifier/regressor

should benchmark again sklearn algorithms on titanic and diabetes dataset
to see efficacy

Functionality:
    should contain hyperparameters
    - max tree depth
    - minimum samples to split node
    - max features for each split
    - should return time needed to train model

    should be able to fit to any generic var;response dataset and
    evaluate both classification/regression performance


Usage design: similar to sklearn
    model = DecisionTree()
    model.fit(x_train, y_train)
    perf = model.evaluate(x_test, y_test)
    time = model.train_time()

    evaluate function should return appropriate performance metrics
"""

import numpy as np
import pandas as pd


class DecisionTree:
    """
    Decision Tree classifier/regressor

    Attributes
    ----------
    model_type: string
        Initialze model as classifier or regressor

    max_depth: int or None
        Maximum depth of tree

    max_features_split: int or None
        Number of features to consider in node split

    min_sample_split: int
        Minimum number of samples in node to be split

    min_sample_leaf: int
        Minimum number of samples covered by leaf to be split

    Methods
    -------
    """

    def __init__(
        self,
        model_type: str,
        max_depth: int = None,
        max_features_split: int = None,
        min_sample_split: int = 2,
        min_sample_leaf: int = 1,
    ):
        """
        Initializes DecisionTree class with default sklearn attributes

        Returns
        -------
            DecisionTree class object

        Raises
        ------
        TypeError
            If model_type not specified during initialization
        """
        self.model_type = model_type
        self.max_depth = max_depth
        self.max_features_split = max_features_split
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf

    def train(self, dataset, target):
        """
        Main method for training model

        Attributes
        ----------
        dataset: pandas dataframe
        target: string
            name of target column in dataset
        """
        pass

    def gini_impurity(self, dataset, target):
        """
        Calculates gini impurity of all features in dataset

        Attributes
        ----------
        dataset: pandas dataframe
            dataframe containing all features & target
        target: string
            column name of target

        Returns
        -------
        Dataframe containing feature and threshold of best split
        """
        impurities = []
        target_np = np.array(dataset[target])
        num_target = len(dataset[target])

        # calculate gini impurity for each feature, append
        # (feature name, min impurity, min impurity index) to impurities list
        features = list(dataset.drop([target], axis=1))
        for feature in features:
            feature_val = np.array(dataset[feature])
            possible_thresholds = np.unique(feature_val)
            feature_impurity = []
            # for each threa
            for threshold in possible_thresholds:
                selection = feature_val >= threshold
                right = target_np[selection]
                left = target_np[~selection]

                num_right = right.size

                _, right_dist = np.unique(right, return_counts=True)
                _, left_dist = np.unique(left, return_counts=True)
                gini_right = 1 - np.sum((np.array(right_dist) / num_right) ** 2)
                gini_left = 1 - np.sum(
                    (np.array(left_dist) / (num_target - num_right)) ** 2
                )

                gini_split = (
                    num_right * gini_right +
                    (num_target - num_right) * gini_left
                ) / num_target
                feature_impurity.append(gini_split)

            impurities.append(
                [min(feature_impurity), feature_impurity.index(
                    min(feature_impurity))]
            )

        # convert gini data to pandas dataframe for easier handling
        df = pd.DataFrame(impurities)
        df = df.transpose()
        df.columns = features
        df.index = ["min gini", "min gini idx"]
        df = df.transpose()

        return df.loc[df["min gini"].idxmin()]

    def get_best_split(self, dataset, target):
        """
        1. load & properly format dataset
        2. Feed dataset into gini_impurity function
        3. Split dataset at threshold from gini_impurity()
        4. Store nodes
        5. Recalculate get_best_split on new dataset
        """
        pass

    def predict(self, predict_features):
        '''
        Should mirror sklearn predict function - take in prediction features and outputs
        prediction
        '''
        pass


data = pd.read_csv("wine.csv")

tree = DecisionTree("regressor", max_depth=10, min_sample_split=5)
print(tree.gini_impurity(data, "quality"))
