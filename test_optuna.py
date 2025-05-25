import unittest
import pandas as pd
import numpy as np
import optuna
from unittest.mock import patch, MagicMock
from lightgbm import LGBMRegressor
from test import optuna_feature_selection_and_hyperparam_objective, rmsle

# Import the function to test


class TestOptunaObjective(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple DataFrame with the required structure
        np.random.seed(42)
        n_samples = 100
        
        # Basic data
        self.df = pd.DataFrame({
            'center_id': np.random.randint(1, 5, n_samples),
            'meal_id': np.random.randint(1, 10, n_samples),
            'week': np.random.randint(1, 53, n_samples),
            'num_orders': np.random.randint(1, 100, n_samples),
            # Required features for objective function
            'checkout_price': np.random.uniform(10, 100, n_samples),
            'base_price': np.random.uniform(15, 120, n_samples),
            'homepage_featured': np.random.choice([0, 1], n_samples),
            'emailer_for_promotion': np.random.choice([0, 1], n_samples),
            'discount': np.random.uniform(0, 20, n_samples),
            'discount_pct': np.random.uniform(0, 0.2, n_samples),
            'price_diff': np.random.uniform(-10, 10, n_samples),
            'center_orders_mean': np.random.uniform(20, 60, n_samples),
            'meal_orders_mean': np.random.uniform(20, 60, n_samples),
            'mean_orders_by_weekofyear': np.random.uniform(20, 60, n_samples),
            'mean_orders_by_month': np.random.uniform(20, 60, n_samples),
            # Add required rolling/lag features
            'num_orders_lag_1': np.random.uniform(10, 90, n_samples),
            'num_orders_lag_2': np.random.uniform(10, 90, n_samples),
            'num_orders_rolling_mean_3': np.random.uniform(20, 80, n_samples),
            'num_orders_rolling_std_3': np.random.uniform(1, 10, n_samples),
            'weekofyear_sin': np.random.uniform(-1, 1, n_samples),
            'weekofyear_cos': np.random.uniform(-1, 1, n_samples),
            'month_sin': np.random.uniform(-1, 1, n_samples),
            'month_cos': np.random.uniform(-1, 1, n_samples),
        })
        
        # Add rolling means for binary features
        for feature in ['emailer_for_promotion', 'homepage_featured']:
            for window in [1, 2, 3, 5, 7, 14, 21, 28]:
                self.df[f'{feature}_rolling_mean_{window}'] = np.random.uniform(0, 1, n_samples)
        
        # Set categorical columns
        for col in ['center_id', 'meal_id']:
            self.df[col] = self.df[col].astype('category')
        
        # Create a MagicMock for the trial
        self.trial = MagicMock()
        
        # Set up the suggest_* methods to return appropriate values
        def suggest_categorical_impl(name, choices):
            if name in self.df.columns:
                return True  # Select all features for testing
            return choices[0] if choices else None
            
        self.trial.suggest_categorical.side_effect = suggest_categorical_impl
        self.trial.suggest_float.return_value = 0.1
        self.trial.suggest_int.return_value = 10
        
        # Set up study property
        self.trial.study = MagicMock()
        self.trial.study.directions = ["minimize"]  # Single objective by default
        
        # Dictionary to store user_attrs
        self.user_attrs = {}
        def set_user_attr_impl(key, value):
            self.user_attrs[key] = value
        self.trial.set_user_attr.side_effect = set_user_attr_impl
        
        # Add user_attrs as a property of trial for retrieval
        type(self.trial).user_attrs = property(lambda self: self.user_attrs)

    @patch('test.LGBMRegressor')
    @patch('test.ExpandingGroupTimeSeriesSplit')
    def test_basic_functionality(self, mock_split_class, mock_lgbm):
        """Test that the function runs without errors and returns expected values."""
        # Set up the mock split class
        mock_split_instance = mock_split_class.return_value
        mock_split_instance.split.return_value = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
        
        # Set up the LGBMRegressor mock
        mock_model = mock_lgbm.return_value
        mock_model.predict.return_value = np.array([10, 20, 30])
        
        # Run the function
        result = optuna_feature_selection_and_hyperparam_objective(self.trial, self.df)
        
        # Check the result
        self.assertIsInstance(result, float)
        
        # Verify the mock was called correctly
        mock_lgbm.assert_called()
        mock_model.fit.assert_called()
        
        # Check user attributes
        self.assertIn('mean_train', self.user_attrs)
        self.assertIn('n_features', self.user_attrs)
        self.assertEqual(self.user_attrs['objective'], result)
    
    @patch('test.LGBMRegressor')
    @patch('test.ExpandingGroupTimeSeriesSplit')
    def test_multi_objective(self, mock_split_class, mock_lgbm):
        """Test multi-objective behavior."""
        # Set up for multi-objective
        self.trial.study.directions = ["minimize", "minimize", "minimize", "minimize"]
        
        # Set up the mock split class
        mock_split_instance = mock_split_class.return_value
        mock_split_instance.split.return_value = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
        
        # Set up the LGBMRegressor mock
        mock_model = mock_lgbm.return_value
        mock_model.predict.return_value = np.array([10, 20, 30])
        
        # Run the function
        result = optuna_feature_selection_and_hyperparam_objective(self.trial, self.df)
        
        # Check the result
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)  # 4 objectives
        
        # Check multi-objective specific user attributes
        self.assertIn('mean_valid', self.user_attrs)
        self.assertIn('generalization_gap', self.user_attrs)
        self.assertIn('gap_penalty', self.user_attrs)
        self.assertIn('complexity_penalty', self.user_attrs)
        self.assertIn('reg_reward', self.user_attrs)

    @patch('test.logging')
    def test_pruning_with_too_few_features(self, mock_logging):
        """Test that trials with too few features are pruned."""
        # Override suggest_categorical to always return False for features
        self.trial.suggest_categorical.side_effect = lambda name, choices: False
        
        # Expect a pruning exception
        with self.assertRaises(optuna.TrialPruned):
            optuna_feature_selection_and_hyperparam_objective(self.trial, self.df)
        
        # Verify warning was logged
        mock_logging.warning.assert_called()


if __name__ == '__main__':
    unittest.main()