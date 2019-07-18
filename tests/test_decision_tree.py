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
        len_target = 1599
        test_data = pd.DataFrame(data[['alcohol', 'quality']])
        tree = DecisionTree('regressor')
        gini_df = tree.calculate_split_gini(data, 'quality', len_target)
        self.assertAlmostEqual(gini_df, 0.8317775184338124)

        data['alcohol'] = 1
        test_data2 = pd.DataFrame(data[['alcohol', 'quality']])
        tree2 = DecisionTree('regressor')
        gini_df2 = tree.calculate_split_gini(data, 'quality', len_target)
        self.assertAlmostEqual(gini_df2, 0.8436599449503852)


if __name__ == '__main__':
    unittest.main(exit=False)
