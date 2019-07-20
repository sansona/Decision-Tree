import unittest
import pandas as pd
from decision_tree import DecisionNode, DecisionTree

data = pd.read_csv("wine.csv")


class TestDecisionNode(unittest.TestCase):
    def test_decision_node_init(self):
        with self.assertRaises(TypeError):
            # init missing all args
            node = DecisionNode()
            node.return_valid_init()

        with self.assertRaises(TypeError):
            # init missing 3 positional args
            node = DecisionNode(4)
            node.return_valid_init()

        with self.assertRaises(TypeError):
            # insert_node w/ missing args
            test_node = DecisionNode('test_node', 1, 1, 1)
            test_node.insert_node('test_child')
        with self.assertRaises(TypeError):

            # insert node w/ inappropriate args
            test_node = DecisionNode('test_node', 1, 1, 1)
            test_node.insert_node('test_child', 1, 1, 'str_gini')


class TestDecisionTree(unittest.TestCase):
    def test_decision_tree_init(self):
        with self.assertRaises(TypeError):
            # init missing args
            DecisionTree()

        with self.assertRaises(ValueError):
            # init with inappropirate model_type arg
            tree = DecisionTree('dummy')
            tree.return_valid_init()

        with self.assertRaises(TypeError):
            # init with inappropriate max_depth arg
            tree = DecisionTree('regressor', max_depth='dummy')
            tree.return_valid_init()

        with self.assertRaises(TypeError):
            # init with inapproprpiate min_sample_leaf arg
            tree = DecisionTree('regressor', min_sample_leaf=0)
            tree.return_valid_init()

        with self.assertRaises(TypeError):
            # init with inapproprpiate max_features_split arg
            tree = DecisionTree('regressor', max_features_split=-2)
            tree.return_valid_init()

    def test_gini_impurity(self):
        # test if gini impurity calculates correct values
        # for cases where impurity is minimized
        data['alcohol'] = 0
        len_target = 1599
        test_data = pd.DataFrame(data[['alcohol', 'quality']])
        tree = DecisionTree('regressor')
        gini_df = tree.calculate_split_gini(data, 'quality', len_target)
        self.assertAlmostEqual(gini_df, 0.832)

        data['alcohol'] = 1
        test_data2 = pd.DataFrame(data[['alcohol', 'quality']])
        tree2 = DecisionTree('regressor')
        gini_df2 = tree.calculate_split_gini(data, 'quality', len_target)
        self.assertAlmostEqual(gini_df2, 0.844)


if __name__ == '__main__':
    unittest.main(exit=False)
