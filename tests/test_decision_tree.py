import unittest
from decision_tree import DecisionTree


class TestDecisionTree(unittest.TestCase):
    def test_decision_tree_init(self):
        # 1. test if class initializes w/o problem
        # 2. test if class fails w/ wrong modeltype input
        # 3. test if class fails w/ wrong attribute types
        pass

    def test_gini_impurity(self):
        # test if gini impurity calculates correct values
        # for cases where impurity = 0.0/0.5/1.0
        pass
