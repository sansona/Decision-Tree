"""
From "scratch" implementation of decision tree classifier/regressor

Usage design: similar to sklearn
    model = DecisionTree("classifier")
    model.train(features, predictor)
"""

from dataclasses import dataclass
import numpy as np

ARBITRARY_LARGE_N = 999999999


@dataclass
class DecisionNode:
    """
    Base node class for decision tree

    Initialize w/ multiple args instead of general data param
    for ease of later access

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

    position: int
        Keeps track of node position in tree

    left_child: DecisionNode
        Node corresponding to values below threshold

    right_child: DecisionNode
        Node corresponding to values above threshold

    Methods
    -------
    return_valid_init:
        Tests whether init arguments adhere to initialization restrictions

    insert_node:
        Grow tree by comparing gini values

    in_tree:
        Recursively traverses tree to determine if node exists in tree

    display_tree:
        Displays entire tree by recursively traversing tree
    """

    feature: str
    threshold: float
    idx: int
    gini: float
    position: int = 0
    left_child = None
    right_child = None

    def return_valid_init(self):
        """
        Tests whether init arguments adhere to initialization restrictions
        """
        assert isinstance(self.feature, str)
        assert isinstance(self.threshold, float)
        assert isinstance(self.idx, int)
        assert isinstance(self.gini, float)

    def insert_node(self, feature, threshold, idx, gini):
        """
        Method for growing tree via. comparing gini values
        Inspired by: https://www.tutorialspoint.com/python/python_binary_tree

        Arguments
        ---------
        See docstring of DecisionNode
        """
        if gini < self.gini:
            if self.left_child is None:
                self.left_child = DecisionNode(feature, threshold, idx, gini)
            else:
                self.left_child.insert_node(feature, threshold, idx, gini)
        elif gini > self.gini:
            if self.right_child is None:
                self.right_child = DecisionNode(feature, threshold, idx, gini)
            else:
                self.right_child.insert_node(feature, threshold, idx, gini)

    def in_tree(self, node):
        """
        Recursively traverses tree to determine if node exists in tree
        """
        if node in (self.left_child, self.right_child):
            return True

        self.in_tree(self.left_child)
        self.in_tree(self.right_child)

        return False

    def display_tree(self):
        """
        Displays entire tree by recursively traversing tree
        """
        print(f"{self.position} {self.feature} : {self.threshold}")
        if self.left_child is not None:
            self.left_child.display_tree()
        if self.right_child is not None:
            self.right_child.display_tree()


@dataclass
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
    return_valid_init:
        Tests whether init arguments adhere to initialization restrictions
    calculate_split_gini:
        Returns gini calculation for feature split

    get_best_gini_split:
        Calculates gini impurity of all features in dataset. Evaluates
        all possible thresholds for all features & values in dataset and
        returns best split conditions. Used as cost function in classifier

    train:
        Recursively generates decision tree determining
        best splits
    """

    model_type: str
    max_depth = None
    max_features_split = None
    min_sample_split = None
    min_sample_leaf: int = 1
    curr_nodes: int = 0
    root = None

    def return_valid_init(self):
        """
        Tests whether init arguments adhere to initialization restrictions
        """
        if self.model_type not in ["classifier", "regressor"]:
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
        gini_split = float(f"{gini_split:.3f}")

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
        min_impurity_dummy = ARBITRARY_LARGE_N
        for node in nodes:
            if node.gini < min_impurity_dummy:
                min_impurity_dummy = node.gini
                best_gini_split = node

        return best_gini_split

    def train(self, dataset, target):
        """

        Trains DecisionTree recursively generates decision
        tree determining best splits

        Attributes
        ----------
        dataset: pandas dataframe
            Dataframe containing all features & target

        target: string
            Column name of target

        Returns
        -------
        Trained DecisionTree object consisting of connected
        DecisionNodes
        """
        # if initialize with no specified max_depth, let tree grow to
        # theoretical max depth. Use dummy value for theoretical max
        if self.max_depth is None:
            depth = ARBITRARY_LARGE_N
        else:
            depth = self.max_depth

        while self.curr_nodes < 2 * depth + 1:
            # get split details for current dataset
            if self.curr_nodes == 0:
                # initialize root node
                self.root = self.get_best_gini_split(dataset, target)
                self.root.position = self.curr_nodes
                feature, thresh_idx = self.root.feature, self.root.idx
                self.curr_nodes += 1
            else:
                node = self.get_best_gini_split(dataset, target)
                node.position = self.curr_nodes
                feature, thresh_idx = node.feature, node.idx

                # insert all new nodes to root node
                if not self.root.in_tree(node):
                    self.root.insert_node(
                        node.feature, node.threshold, node.idx, node.gini
                    )
                    self.curr_nodes += 1
                else:
                    # terminate infinite loop when min_gini accomplished as
                    # curr_depth < max_depth
                    break

            # thresh_idx == 0 corresponds to a split with no decrease
            # in impurity. Thus, as long as idx !=, continue splitting
            if thresh_idx != 0:
                # sort values prior to recalling function to split
                # @ appropriate index
                df = dataset.sort_values(feature)
                left = df[:thresh_idx]
                right = df[thresh_idx:]
                self.train(left, target)
                self.train(right, target)

        return self.root
