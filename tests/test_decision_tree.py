import unittest
import pandas as pd
from decision_tree import DecisionTree

data = pd.read_csv("wine.csv")


class TestDecisionTree(unittest.TestCase):
    def test_decision_tree_init(self):
        with self.assertRaises(TypeError):
            DecisionTree()
        with self.assertRaises(ValueError):
            DecisionTree('dummy')
        with self.assertRaises(TypeError):
            DecisionTree('regressor', max_depth='dummy')
        with self.assertRaises(TypeError):
            DecisionTree('regressor', min_sample_leaf=0)
        with self.assertRaises(TypeError):
            DecisionTree('regressor', max_features_split=-2)

    def test_gini_impurity(self):
        # test if gini impurity calculates correct values
        # for cases where impurity is minimized
        data['alcohol'] = 0
        test_data = pd.DataFrame(data[['alcohol', 'quality']])
        tree = DecisionTree('regressor')
        gini_df = tree.gini_impurity(data, 'quality')
        self.assertAlmostEqual(gini_df["threshold"], 0.56)

        data['alcohol'] = 1
        test_data2 = pd.DataFrame(data[['alcohol', 'quality']])
        tree2 = DecisionTree('regressor')
        gini_df2 = tree.gini_impurity(data, 'quality')
        self.assertAlmostEqual(gini_df2["threshold"], 0.56)


if __name__ == '__main__':
    unittest.main(exit=False)
