import pandas as pd
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # ==== 1. Load the dataset ====
    # just reading the steel.csv file from the data folder we created earlier
    df = pd.read_csv("data/steel.csv")

    # X = all input features, y = target (tensile_strength)
    X = df.drop("tensile_strength", axis=1)
    y = df["tensile_strength"]

    print("Data shape:", df.shape)
    print(df.head(), "\n")

    # ==== 2. Set up 10-fold cross validation ====
    # shuffle=True so folds are mixed up, random_state so results are reproducible
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # We'll use these metrics for everything:
    scoring = ["neg_mean_squared_error", "r2"]

    # =====================================================================
    #                DECISION TREE REGRESSOR (BASELINE)
    # =====================================================================
    print("=" * 70)
    print("Decision Tree Regressor - default hyperparameters")
    print("=" * 70)

    dt_default = DecisionTreeRegressor(random_state=42)

    dt_default_scores = cross_validate(
        dt_default,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    # cross_validate returns *negative* MSE for minimization, so flip sign
    dt_train_mse = -dt_default_scores["train_neg_mean_squared_error"].mean()
    dt_test_mse = -dt_default_scores["test_neg_mean_squared_error"].mean()
    dt_train_r2 = dt_default_scores["train_r2"].mean()
    dt_test_r2 = dt_default_scores["test_r2"].mean()

    print(f"Train MSE: {dt_train_mse:.4f}")
    print(f"Test  MSE: {dt_test_mse:.4f}")
    print(f"Train R²: {dt_train_r2:.4f}")
    print(f"Test  R²: {dt_test_r2:.4f}\n")

    # =====================================================================
    #          DECISION TREE REGRESSOR - HYPERPARAMETER TUNING
    # =====================================================================
    print("=" * 70)
    print("Decision Tree Regressor - GridSearchCV (tuning)")
    print("=" * 70)

    # grid for the two hyperparameters we talked about in the report
    dt_param_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    }

    dt = DecisionTreeRegressor(random_state=42)

    dt_grid = GridSearchCV(
        estimator=dt,
        param_grid=dt_param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",  # optimise for MSE
        return_train_score=True,
        n_jobs=-1,
    )

    # fit on the whole dataset (GridSearchCV will handle the folds internally)
    dt_grid.fit(X, y)

    print("Best params (DT):", dt_grid.best_params_)
    print("Best CV score (negative MSE):", dt_grid.best_score_)

    # save the whole grid search table so we can make graphs/tables later
    dt_results_df = pd.DataFrame(dt_grid.cv_results_)
    dt_results_df.to_csv("dt_grid_results.csv", index=False)
    print("Full Decision Tree tuning results saved to dt_grid_results.csv\n")

    # now evaluate the *best* DT model more cleanly with both MSE + R²
    dt_best = dt_grid.best_estimator_

    dt_best_scores = cross_validate(
        dt_best,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    dt_best_train_mse = -dt_best_scores["train_neg_mean_squared_error"].mean()
    dt_best_test_mse = -dt_best_scores["test_neg_mean_squared_error"].mean()
    dt_best_train_r2 = dt_best_scores["train_r2"].mean()
    dt_best_test_r2 = dt_best_scores["test_r2"].mean()

    print("Decision Tree (best tuned model):")
    print(f"Train MSE: {dt_best_train_mse:.4f}")
    print(f"Test  MSE: {dt_best_test_mse:.4f}")
    print(f"Train R²: {dt_best_train_r2:.4f}")
    print(f"Test  R²: {dt_best_test_r2:.4f}\n")

    # =====================================================================
    #                  SVR (WITH SCALING) - BASELINE
    # =====================================================================
    print("=" * 70)
    print("SVR (RBF kernel) - default hyperparameters")
    print("=" * 70)

    # SVR is sensitive to feature scales, so I’m just wrapping it in a pipeline
    svr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf")),
        ]
    )

    svr_default_scores = cross_validate(
        svr_pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    svr_train_mse = -svr_default_scores["train_neg_mean_squared_error"].mean()
    svr_test_mse = -svr_default_scores["test_neg_mean_squared_error"].mean()
    svr_train_r2 = svr_default_scores["train_r2"].mean()
    svr_test_r2 = svr_default_scores["test_r2"].mean()

    print(f"Train MSE: {svr_train_mse:.4f}")
    print(f"Test  MSE: {svr_test_mse:.4f}")
    print(f"Train R²: {svr_train_r2:.4f}")
    print(f"Test  R²: {svr_test_r2:.4f}\n")

    # =====================================================================
    #                     SVR - HYPERPARAMETER TUNING
    # =====================================================================
    print("=" * 70)
    print("SVR (RBF kernel) - GridSearchCV (tuning C and epsilon)")
    print("=" * 70)

    # note: because we’re using a Pipeline, we have to prefix params with "svr__"
    svr_param_grid = {
        "svr__C": [0.1, 1, 10, 100],
        "svr__epsilon": [0.01, 0.1, 0.2, 0.5],
    }

    svr_grid = GridSearchCV(
        estimator=svr_pipeline,
        param_grid=svr_param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        n_jobs=-1,
    )

    svr_grid.fit(X, y)

    print("Best params (SVR):", svr_grid.best_params_)
    print("Best CV score (negative MSE):", svr_grid.best_score_)

    svr_results_df = pd.DataFrame(svr_grid.cv_results_)
    svr_results_df.to_csv("svr_grid_results.csv", index=False)
    print("Full SVR tuning results saved to svr_grid_results.csv\n")

    # evaluate the best SVR model properly with both metrics
    svr_best = svr_grid.best_estimator_

    svr_best_scores = cross_validate(
        svr_best,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    svr_best_train_mse = -svr_best_scores["train_neg_mean_squared_error"].mean()
    svr_best_test_mse = -svr_best_scores["test_neg_mean_squared_error"].mean()
    svr_best_train_r2 = svr_best_scores["train_r2"].mean()
    svr_best_test_r2 = svr_best_scores["test_r2"].mean()

    print("SVR (best tuned model):")
    print(f"Train MSE: {svr_best_train_mse:.4f}")
    print(f"Test  MSE: {svr_best_test_mse:.4f}")
    print(f"Train R²: {svr_best_train_r2:.4f}")
    print(f"Test  R²: {svr_best_test_r2:.4f}\n")

    print("All done. Check dt_grid_results.csv and svr_grid_results.csv for detailed tables.")


if __name__ == "__main__":
    # just calling main so the script runs when we hit the green play button
    main()

