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


class Node:
    def __init__(self, condition):
        # condition should be a dict [feature: threshold]
        self.left = None
        self.right = None
        self.condition = condition

    def insert(self, condition):
        pass

    def printnode(self):
        return self.condition


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
        Calculates gini impurity of all features in dataset. Evaluates
        all possible thresholds for all features & values in dataset and
        returns best threshold

        Attributes
        ----------
        dataset: pandas dataframe
            dataframe containing all features & target. Since this
            function is called from get_best_split(), len(dataset.columns)
            will decrease by 1 everytime it's called
        target: string
            column name of target

        Returns
        -------
        Dataframe containing feature name, best threshold, and gini score
        corresponding to best threshold
        """
        impurities = list()
        target_np = np.array(dataset[target])
        num_target = len(dataset[target])

        # calculate gini impurity for each feature, append
        # (feature name, threshold val, min impurity)
        features = list(dataset.drop([target], axis=1))
        for feature in features:
            feature_val = np.array(dataset[feature])
            possible_thresholds = sorted(feature_val)
            feature_impurity = []  # gini impurities for each sample

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

                # save gini impurity score for threshold
                gini_split = (
                    num_right * gini_right +
                    (num_target - num_right) * gini_left
                ) / num_target
                feature_impurity.append(gini_split)

            # save value, index, and gini score of threshold
            thresh_idx = feature_impurity.index(min(feature_impurity))
            impurities.append(
                [feature_val[thresh_idx], thresh_idx, min(feature_impurity)]
            )

        # convert gini data to pandas dataframe for easier handling
        df = pd.DataFrame(impurities)
        df = df.transpose()
        df.columns = features
        df.index = ["threshold", "thresh idx", "min gini"]
        df = df.transpose()

        return df.loc[df["min gini"].idxmin()]

    def get_best_split(self, dataset, target):
        """
        Calculates best thresholds for entirety of dataset by
        iteratively calling gini_impurity and storing best feature/threshold

        Attributes
        ----------
        dataset: pandas dataframe
            dataframe containing all features & target
        target: string
            column name of target

        Returns
        -------
        Dataframe containing best threshold for each feature
        """
        # if initialize with no specified max_depth, let tree grow to
        # theoretical max depth. Use dummy value for theoretical max
        if self.max_depth == None:
            depth = 9999999
        else:
            depth = self.max_depth

        # while true, find best threshold, then remove threshold feature
        # and repeat until reach max_depth or run out of features

        splits = list()
        curr_depth = 0
        # base case - for first split, feature to split is unknown
        if curr_depth == 0:
            min_gini = self.gini_impurity(dataset, target)
            feature, threshold, thresh_idx = (
                min_gini.name,
                min_gini["threshold"],
                min_gini["thresh idx"],
            )
            splits.append([feature, threshold, thresh_idx])
            curr_depth += 1

        # TODO - this SHOULD iterate through and run gini_impurity on each
        # subset of the dataframe, appending the proper split to some
        # as of now undecided data structure
        df_left = dataset.copy(deep=True)
        df_right = dataset.copy(deep=True)
        while curr_depth <= depth and curr_depth < len(dataset.columns):
            split_feature = splits[-1][0]
            split_idx = int(splits[-1][-1])

            print(split_feature)
            # print(splits)
            # sort values, split dataframe @ thresh index
            df_left.sort_values(split_feature, inplace=True)
            df_right = df_left.iloc[split_idx:]
            df_left = df_left.iloc[:split_idx]
            print(df_left.shape, df_right.shape)
            min_gini_left = self.gini_impurity(df_left, target)
            min_gini_right = self.gini_impurity(df_right, target)

            for gini in [min_gini_left, min_gini_right]:
                feature, threshold, thresh_idx = (
                    gini.name,
                    gini["threshold"],
                    gini["thresh idx"],
                )
                splits.append([feature, threshold, thresh_idx])
            curr_depth += 1

        return splits

        """
        df = pd.DataFrame(splits, columns=["feature", "threshold", "min gini"])
        df.set_index("feature", inplace=True)

        return df
        """

    def predict(self, predict_features):
        """
        Should mirror sklearn predict function - take in prediction features and outputs
        prediction
        """
        pass


data = pd.read_csv("wine.csv")

tree = DecisionTree("regressor", max_depth=7)
print(tree.get_best_split(data, "quality").to_string())
