# Food Delivery Forecast Model Enhancement: Final Report

## Problem Summary
The enhanced model for food delivery order forecasting was reported to be "way off" after switching from RMSE to RMSLE optimization. Our investigation identified key issues and implemented fixes to align the model more closely with the original implementation.

## Root Causes Identified

1. **Recursive Prediction Missing**: The enhanced model didn't use the proper recursive prediction approach crucial for time-series forecasting.

2. **RMSLE Integration Issues**: While the RMSLE metric was correctly implemented, it wasn't properly integrated with LightGBM's training process.

3. **Feature Engineering Bugs**: The recursive prediction process broke due to categorical column handling after one-hot encoding.

4. **Model Objective Mismatch**: The original model used a 'regression_l1' (MAE) objective which often works better with RMSLE than the default objective.

## Key Fixes Implemented

1. Properly implemented recursive prediction for test weeks (predicting one week at a time).
2. Added robust error handling for categorical features during feature engineering.
3. Properly integrated RMSLE with LightGBM using custom eval metrics and early stopping.
4. Fixed parameter optimization to better align with RMSLE objectives.

## Results After Fixes

The enhanced model now shows:
- High correlation with original model (0.964)
- Mean absolute difference of 34.91 orders (16.47%)
- Validation RMSE of 92.89 and RMSLE of 0.5008

## Weekly Trend Analysis

Our weekly analysis reveals interesting patterns:
- **Weeks 146-149**: Enhanced model predicts higher than original (+4.9% to +9.3%)
- **Weeks 150-151**: Both models closely aligned (-1.1% to +0.16%)
- **Weeks 152-155**: Enhanced model predicts lower than original (-3.9% to -14.0%)

This suggests the enhanced model might be capturing different temporal patterns, particularly showing more conservative predictions in later forecast weeks.

## Category-Specific Findings

- **Sandwich category**: Largest differences (138.55 orders, 19.6%)
- **Rice Bowl category**: Second largest differences (109.10 orders, 14.2%)
- **Italian cuisine**: Largest cuisine-specific differences (52.49 orders, 14.9%)

## Center Type Analysis

- **TYPE_B centers**: Highest absolute differences (42.38 orders)
- **TYPE_C centers**: Highest relative differences (18.0%)

## Top 10 Individual Differences

All top differences are in the Sandwich category with Italian cuisine, particularly meal_id 1754 which appears in 8 of the top 10 difference records. This suggests something specific about this meal type may not be captured well in the enhanced model.

## Recommended Next Steps

1. **Immediate Improvements**:
   - Create category-specific features or models, particularly for Sandwich and Rice Bowl categories
   - Implement error correction in the recursive prediction to improve later week forecasts

2. **Medium-term Enhancements**:
   - Develop an ensemble approach combining both models with optimized weights
   - Analyze specific features for Italian cuisine and Sandwich category to improve predictions

3. **Future Research**:
   - Investigate why weeks 152-153 show significantly different predictions
   - Consider specialized models for different center types
   - Explore more sophisticated time series techniques to better capture temporal patterns

## Conclusion

The enhanced model has been successfully fixed and now provides reasonable forecasts that are generally aligned with the original model. The differences observed are largely explained by variations in the feature engineering approach and hyperparameter optimization. 

By implementing the recommended improvements, we can further refine the model to provide even more accurate food delivery order forecasts.
