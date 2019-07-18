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

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DecisionNode:
    """
    Base node class for decision tree

    Attributes:
    -----------
    feature: str
        Name of feature
    threshold: float
        Value of threshold value corresponding to min gini
    idx: int
        Index of threshold value
    gini: float
        Gini impurity corresponding to threshold
    left_child: DecisionNode
        Node corresponding to values below threshold
    right_child: DecisionNode
        Node corresponding to values above threshold
    """

    feature: str
    threshold: float
    idx: int
    gini: float
    left_child = None
    right_child = None


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
    calculate_split_gini:
            Returns gini calculation for feature split

    get_best_gini_split:
            Calculates gini impurity of all features in dataset. Evaluates
            all possible thresholds for all features & values in dataset and
            returns best split conditions. Used as cost function in classifier

    generate_tree:
            Recursively generates decision tree determining
            best splits
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
                If no value for model_type is passed
        ValueError
                If value for model_type not classifier or regressor
        """
        self.model_type = model_type
        self.max_depth = max_depth
        self.max_features_split = max_features_split
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.curr_nodes = 0
        self.splits = list()

        # quick checks to ensure object initiated with arguments
        # of proper types
        if model_type not in ["classifier", "regressor"]:
            raise ValueError("Missing arg: model_type")
        for var in [
            self.max_depth,
            self.max_features_split,
            self.min_sample_split,
            self.min_sample_leaf,
        ]:
            if not isinstance(var, int) and var is not None:
                raise TypeError(f"{var} of wrong type")
            if var is not None and var < 1:
                raise TypeError(f"{var} cannot be a negative number")

    def calculate_split_gini(self, right, left, len_target):
        """
        returns gini calculation for feature split

        Args
        ----
        right: nupmy array
                Side of dataset split at value
        left: numpy array
                Remaining side of dataset split at value
        len_target: int
                Number of datapoints in target column

        Returns
        -------
        Float corresponding to gini impurity of right/left split

        """
        _, right_dist = np.unique(right, return_counts=True)
        _, left_dist = np.unique(left, return_counts=True)

        len_right = right.size

        gini_right = 1 - np.sum((np.array(right_dist) / len_right) ** 2)
        gini_left = 1 - np.sum((np.array(left_dist) /
                                (len_target - len_right)) ** 2)

        # weighted sum of gini of individual splits
        gini_split = (
            len_right * gini_right + (len_target - len_right) * gini_left
        ) / len_target

        return gini_split

    def get_best_gini_split(self, dataset, target):
        """
        Calculates gini impurity of all features in dataset. Evaluates
        all possible thresholds for all features & values in dataset and
        returns best split conditions. Used as cost function in classifier

        Attributes
        ----------
        dataset: pandas dataframe
                Dataframe containing all features & target

        target: string
                Column name of target

        Returns
        -------
        DecisionNode corresponding to feature split that minimzes gini
        """
        nodes = list()
        target_np_array = np.array(dataset[target])
        len_target = len(dataset[target])

        # calculate gini impurity for each possible split
        features = list(dataset.drop([target], axis=1))
        for feature in features:
            feature_val = np.array(dataset[feature])
            possible_thresholds = sorted(feature_val)
            feature_impurity = list()  # gini impurities for each sample

            for value in possible_thresholds:
                selection = feature_val >= value
                right = target_np_array[selection]
                left = target_np_array[~selection]

                gini_split = self.calculate_split_gini(right, left, len_target)
                feature_impurity.append(gini_split)

            # save value, index, and gini score of split with minimum impurity

            min_impurity = min(feature_impurity)
            min_gini_idx = feature_impurity.index(min_impurity)
            decision = DecisionNode(
                feature, feature_val[min_gini_idx], min_gini_idx, min_impurity
            )

            nodes.append(decision)

        # iterate through nodes, return node corresponding to minimum gini
        min_impurity_dummy = 999999
        for node in nodes:
            if node.gini < min_impurity_dummy:
                min_impurity_dummy = node.gini
                best_gini_split = node

        return best_gini_split

    def get_best_mse_split(self, dataset, target):
        """
        Calculates mean squared error of all features in dataset. Evaluates
        all possible thresholds for all features & values in dataset and
        returns best split conditions. Used as cost function in regressor

        Attributes
        ----------
        dataset: pandas dataframe
                Dataframe containing all features & target

        target: string
                Column name of target

        Returns
        -------
        Dataframe containing feature name, best threshold, and gini score
        corresponding to best threshold
        """
        pass

    def generate_tree(self, dataset, target):
        """
        Recursively generates decision tree determining
        best splits

        Attributes
        ----------
        dataset: pandas dataframe
                Dataframe containing all features & target

        target: string
                Column name of target

        Returns
        -------
        List of dictionaries of form:
                {Feature -> str: [threshold, threshold_idx] -> list}
        where each dictionary is a node
        """
        # if initialize with no specified max_depth, let tree grow to
        # theoretical max depth. Use dummy value for theoretical max
        if self.max_depth is None:
            depth = 9999999
        else:
            depth = self.max_depth

        while self.curr_nodes < 2 * depth + 1:
            # get split details for current dataset
            decision = self.get_best_gini_split(dataset, target)
            feature, threshold, thresh_idx = (
                decision.feature,
                decision.threshold,
                decision.idx,
            )
            # append split conditions to self.split if condition not
            # already there
            # TODO: cleanup loop execution
            # TODO: self.splits could be a better datatype
            if decision not in self.splits:
                self.splits.append(decision)
                self.curr_nodes += 1
            else:
                break

            # thresh_idx == 0 corresponds to a split with no decrease
            # in impurity. Thus, as long as idx !=, continue splitting
            if thresh_idx != 0:
                # sort values prior to recalling function to split
                # @ appropriate index
                df = dataset.sort_values(feature)
                left = df[:thresh_idx]
                right = df[thresh_idx:]
                self.generate_tree(left, target)
                self.generate_tree(right, target)
            else:
                continue

        # TODO - need to make sure this is doing what it should be doing
        return self.splits

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

    def predict(self, predict_features):
        """
        Should mirror sklearn predict function - take in prediction features and outputs
        prediction
        """
        pass


'''
data = pd.read_csv("wine.csv")

tree = DecisionTree("regressor", max_depth=3)
print(tree.generate_tree(data, "quality"))
'''
