"""
Train the dual-camera 3D position regressor.

This script converts paired CSV data (collected via executor.py save_csv_yolo)
into the three-regressor model {reg_x, reg_y, reg_z} saved as a joblib .pkl file.

Feature vector (14-dim):
    cls_id, px1, py1, w1, h1, conf1,
    px2, py2, w2, h2, conf2,
    ee_x, ee_y, ee_z

Labels: world_x, world_y, world_z (three separate single-output regressors)

Usage:
    python train_regressor.py --data_csv yolo_data.csv --output hanoi_regressor.pkl
    python train_regressor.py --data_csv yolo_data.csv --output hanoi_regressor.pkl \
        --model_type xgboost --test_size 0.2
"""
import argparse
import numpy as np
import pandas as pd
import joblib


def load_data(csv_path):
    """Load paired data from CSV. Returns X (features), y (labels), df (full)."""
    df = pd.read_csv(csv_path)

    feature_cols = [
        "cls", "px_cam1", "py_cam1", "w_cam1", "h_cam1", "conf_cam1",
        "px_cam2", "py_cam2", "w_cam2", "h_cam2", "conf_cam2",
        "ee_x", "ee_y", "ee_z",
    ]

    X = df[feature_cols].values.astype(np.float64)
    y_x = df["world_x"].values.astype(np.float64)
    y_y = df["world_y"].values.astype(np.float64)
    y_z = df["world_z"].values.astype(np.float64)

    return X, y_x, y_y, y_z, df, feature_cols


def make_regressor(model_type, **kwargs):
    """Factory for sklearn-compatible regressors."""
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", None),
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_and_evaluate(X, y_x, y_y, y_z, model_type, test_size):
    """Train three regressors and report MAE."""
    from sklearn.model_selection import train_test_split

    if test_size > 0:
        X_train, X_test, yx_train, yx_test, yy_train, yy_test, yz_train, yz_test = \
            train_test_split(X, y_x, y_y, y_z, test_size=test_size, random_state=42)
    else:
        X_train, yx_train, yy_train, yz_train = X, y_x, y_y, y_z
        X_test = yx_test = yy_test = yz_test = None

    reg_x = make_regressor(model_type)
    reg_y = make_regressor(model_type)
    reg_z = make_regressor(model_type)

    reg_x.fit(X_train, yx_train)
    reg_y.fit(X_train, yy_train)
    reg_z.fit(X_train, yz_train)

    print(f"Training completed ({model_type}, {X_train.shape[0]} samples)")

    if X_test is not None:
        pred_x = reg_x.predict(X_test)
        pred_y = reg_y.predict(X_test)
        pred_z = reg_z.predict(X_test)

        mae_x = np.mean(np.abs(pred_x - yx_test))
        mae_y = np.mean(np.abs(pred_y - yy_test))
        mae_z = np.mean(np.abs(pred_z - yz_test))
        euclidean_errors = np.sqrt(
            (pred_x - yx_test) ** 2 + (pred_y - yy_test) ** 2 + (pred_z - yz_test) ** 2
        )
        mean_euclidean = np.mean(euclidean_errors)

        print(f"Test set: {X_test.shape[0]} samples")
        print(f"  MAE  X: {mae_x:.4f} m  ({mae_x*100:.2f} cm)")
        print(f"  MAE  Y: {mae_y:.4f} m  ({mae_y*100:.2f} cm)")
        print(f"  MAE  Z: {mae_z:.4f} m  ({mae_z*100:.2f} cm)")
        print(f"  Mean Euclidean error: {mean_euclidean:.4f} m  ({mean_euclidean*100:.2f} cm)")

        return reg_x, reg_y, reg_z, {"mae_x": mae_x, "mae_y": mae_y, "mae_z": mae_z,
                                      "euclidean": mean_euclidean}
    return reg_x, reg_y, reg_z, None


def main():
    parser = argparse.ArgumentParser(
        description="Train dual-camera 3D position regressor"
    )
    parser.add_argument("--data_csv", required=True,
                        help="Path to CSV from save_csv_yolo()")
    parser.add_argument("--output", default="regressor.pkl",
                        help="Output .pkl file path")
    parser.add_argument("--model_type", default="random_forest",
                        choices=["random_forest", "xgboost", "gradient_boosting"],
                        help="Regression model type")
    parser.add_argument("--n_estimators", type=int, default=200,
                        help="Number of trees / boosting rounds")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Max tree depth (None = unlimited for RF)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split fraction (0 = train on all data)")
    args = parser.parse_args()

    X, y_x, y_y, y_z, df, feature_cols = load_data(args.data_csv)
    print(f"Loaded {len(df)} samples, {len(feature_cols)} features")
    print(f"Feature columns: {feature_cols}")

    reg_x, reg_y, reg_z, metrics = train_and_evaluate(
        X, y_x, y_y, y_z, args.model_type, args.test_size
    )

    models = {"reg_x": reg_x, "reg_y": reg_y, "reg_z": reg_z}
    joblib.dump(models, args.output)
    print(f"Model saved to {args.output}")

    if metrics and metrics["euclidean"] > 0.05:
        print(
            "\nWARNING: Mean Euclidean error > 5 cm. Consider:\n"
            "  - Collecting more training data covering the full workspace\n"
            "  - Trying --model_type xgboost for stronger non-linear fitting\n"
            "  - Checking camera configurations provide sufficient parallax\n"
        )


if __name__ == "__main__":
    main()
