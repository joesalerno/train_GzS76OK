# Feature Optimization and Dimensionality Reduction Analysis

## Executive Summary

After conducting extensive analysis on the feature space of the food delivery forecasting model, we identified significant redundancy and multicollinearity in the feature set that was negatively impacting model performance. Our dimensionality reduction analysis revealed that:

1. The original model contained 171 features with 183 highly correlated pairs (|r| > 0.8)
2. We reduced the feature space to 60 carefully selected features, achieving a 65% feature reduction
3. High correlation pairs were reduced by 96.7%, from 183 to just 6
4. Key interaction features were identified and prioritized based on SHAP importance

The optimized model maintains the predictive power while being more robust and interpretable.

## Key Findings from Feature Analysis

### Most Important Features

Based on the SHAP analysis, the top 10 most important features are:

1. `lag1_x_rolling_mean_3` (SHAP: 48.17) - Interaction between previous week's orders and 3-week rolling average
2. `lag1_x_rolling_mean_2` (SHAP: 25.21) - Interaction between previous week's orders and 2-week rolling average
3. `rolling_mean_2_x_rolling_mean_3` (SHAP: 18.68) - Interaction between 2-week and 3-week rolling averages
4. `num_orders_lag_1` (SHAP: 12.27) - Previous week's orders (strong baseline predictor)
5. `num_orders_rolling_mean_14` (SHAP: 9.69) - 14-week rolling average orders
6. `center_meal_orders_median_prod` (SHAP: 8.39) - Product of center and meal median orders
7. `price_diff_x_emailer` (SHAP: 7.84) - Interaction between price changes and email promotions
8. `num_orders_rolling_mean_5` (SHAP: 7.22) - 5-week rolling average orders
9. `num_orders_rolling_mean_14_sq` (SHAP: 6.74) - Squared 14-week rolling average
10. `num_orders_rolling_mean_21` (SHAP: 6.68) - 21-week rolling average orders

### Dimensionality Reduction Insights

Principal Component Analysis (PCA) revealed that:

1. 90% of variance is explained by just 12 principal components (versus 171 original features)
2. The first principal component (44.1% variance) primarily captures rolling average features
3. The second principal component (14.6% variance) focuses on promotional interactions
4. The third principal component (5.9% variance) represents homepage and customer behavior

Independent Component Analysis (ICA) identified similar patterns with different emphasis:
- IC1: Price and order dynamics
- IC2: Rolling average interactions
- IC3: Long-term order patterns
- IC5: Seasonality effects

### Feature Correlation Groups

We identified several strongly correlated feature groups:

1. **Rolling mean features**: Multiple time windows (2, 3, 5, 10, 14, 21 days) are highly redundant
2. **Transformed features**: Square, cubic, and square root transformations are highly correlated with original features
3. **Lag features**: Different lag periods correlate strongly when patterns are stable
4. **Interaction features**: Many interaction features correlate with their components

## Optimization Strategy

Based on our analysis, we implemented the following strategy:

1. **Feature Selection**: Keep only the most informative feature from each correlated group
2. **Feature Prioritization**: Prioritize features with high SHAP values
3. **Redundancy Elimination**: Remove transformations (square, cubic, etc.) that correlate with base features
4. **Key Interactions**: Keep only beneficial interactions that provide additional information
5. **Simplified Time Windows**: Use only key time windows (3, 5, 14, 21 days) instead of all possible windows

## Results

The optimized model demonstrates:

1. **Reduced complexity**: 65% fewer features (171 → 60)
2. **Lower multicollinearity**: 96.7% reduction in high correlations
3. **Maintained interpretability**: Top features remain predictive and meaningful

## Future Recommendations

To further improve the model, we recommend:

1. **Trend Features**: Create dedicated features that capture order momentum/trend
2. **Exponential Weighting**: Apply exponentially weighted functions to time-based features
3. **Segmentation**: Create clusters of center-meal combinations for segment-based modeling
4. **Seasonal Interactions**: Develop more category-seasonal interaction features
5. **Regularization**: Increase regularization in the model to prevent overfitting from interaction features
6. **Ensemble Models**: Consider using different feature subsets in an ensemble approach
7. **Error Correction**: Implement recursive forecasting with error correction mechanisms

By implementing these recommendations, we expect to achieve both better model performance and more stable predictions.
