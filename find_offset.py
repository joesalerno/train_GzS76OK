import numpy as np
from lightgbm import LGBMRegressor

def find_best_week_offset(df, features, target, params, holdout_weeks=8):
    # Always split on the original week, not the offset
    max_week = df['week'].max()
    train_mask = df['week'] <= max_week - holdout_weeks
    valid_mask = df['week'] > max_week - holdout_weeks

    best_rmsle = float('inf')
    best_offset = 0

    for offset in range(52):
        df_temp = df.copy()
        # Only create the offset feature, do not touch 'week'
        df_temp['weekofyear_offset'] = (df_temp['week'] + offset) % 52

        # If you have a feature creation function, call it here with use_offset=True
        # df_temp = create_features(df_temp, use_offset=True)

        X_train = df_temp.loc[train_mask, features]
        y_train = df_temp.loc[train_mask, target]
        X_valid = df_temp.loc[valid_mask, features]
        y_valid = df_temp.loc[valid_mask, target]

        model = LGBMRegressor(**params, n_estimators=500)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        rmsle_val = np.sqrt(np.mean((np.log1p(np.clip(preds, 0, None)) - np.log1p(np.clip(y_valid, 0, None))) ** 2))
        print(f"Offset {offset}: Validation RMSLE = {rmsle_val:.5f}")

        if rmsle_val < best_rmsle:
            best_rmsle = rmsle_val
            best_offset = offset

    print(f"\nBest week offset: {best_offset} (Validation RMSLE: {best_rmsle:.5f})")
    return best_offset

if __name__ == "__main__":
    import pandas as pd
    # Load your data
    df = pd.read_csv("train.csv")
    # Example: define features (must include 'weekofyear_offset')
    features = [
        'center_id', 'meal_id', 'checkout_price', 'base_price',
        'homepage_featured', 'weekofyear_offset'
    ]
    # Only keep features that exist in df
    features = [f for f in features if f in df.columns or f == 'weekofyear_offset']
    target = 'num_orders'
    params = dict(objective='regression', random_state=42)
    # Call the function
    find_best_week_offset(df, features, target, params, holdout_weeks=8)