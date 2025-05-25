import pandas as pd
import numpy as np
from recursive_hybrid_forecast import apply_feature_engineering

def main():
    # Minimal test DataFrame with all key columns
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'week': [1, 2, 3],
        'center_id': [10, 10, 10],
        'meal_id': [100, 100, 100],
        'checkout_price': [200.0, 210.0, 220.0],
        'base_price': [250.0, 250.0, 250.0],
        'emailer_for_promotion': [0, 1, 0],
        'homepage_featured': [1, 0, 1],
        'num_orders': [50, 60, 70],
        'category': ['Beverages', 'Beverages', 'Beverages'],
        'cuisine': ['Italian', 'Italian', 'Italian'],
        'center_type': ['TYPE_A', 'TYPE_A', 'TYPE_A']
    })
    print('Before feature engineering:', df.columns.tolist())
    df_eng, weekofyear_means, month_means = apply_feature_engineering(df, is_train=True)
    print('After feature engineering:', df_eng.columns.tolist())
    # Check for presence of key features
    for col in ['meal_id', 'cuisine', 'category', 'center_id', 'center_type']:
        print(f"{col} present after feature engineering? {col in df_eng.columns}")

if __name__ == '__main__':
    main()
