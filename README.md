ML-Assignment2 — Regression using Scikit-Learn

This project was completed as part of CT4101 Machine Learning (University of Galway).
The goal was to build, evaluate, and tune two non-linear regression models using the steel tensile strength dataset.

The models implemented:

Decision Tree Regressor

Support Vector Regressor (SVR) with RBF kernel

Both models were trained using 10-fold Cross-Validation, and performance was measured using:

MSE (Mean Squared Error) — domain-independent metric

R² Score — domain-specific regression metric

All tuning was performed using GridSearchCV, and results were saved to CSV files and visualised with matplotlib.

Project Structure
ML-Assignment2/
│
data/
│   steel.csv                   # Provided dataset
│
main.py                         # Runs models + tuning + CV evaluation
│
dt_graphs.py                    # Generates Decision Tree heatmap + line plot
svr_graphs.py                   # Generates SVR heatmap + line plot
│
dt_grid_results.csv             # GridSearch results (Decision Tree)
svr_grid_results.csv            # GridSearch results (SVR)
│
dt_heatmap.png                  # Saved plot (Decision Tree)
dt_lineplot.png                 # Saved plot (Decision Tree)
svr_heatmap.png                 # Saved plot (SVR)
svr_lineplot.png                # Saved plot (SVR)
│
README.md                       # Project documentation

How to Run the Project
1️. Install Dependencies

Inside the project folder, run:

python -m pip install numpy pandas scikit-learn matplotlib seaborn

2️. Run the Main Model Script
python main.py


This will:

Train both models

Run 10-fold cross-validation

Tune hyperparameters using GridSearchCV

Save all results into CSV files

3. Generate Graphs

Decision Tree plots:

python dt_graphs.py


SVR plots:

python svr_graphs.py


These scripts will create .png files for visualisation.

Model Results Summary
Decision Tree (tuned)

Best params: max_depth=None, min_samples_split=5

Test MSE: 1515.38

Test R²: 0.7976

Strong model but tends to overfit, even after tuning.

SVR (tuned)

Best params: C=100, epsilon=0.5

Test MSE: 1168.87

Test R²: 0.8523

Best overall performance, smoother regression function.

Key Takeaways

Decision Trees capture non-linear structure easily but can overfit quickly.

SVR needs scaling and tuning but delivers far better generalisation.

SVR was the best-performing model for predicting tensile strength.

GridSearchCV significantly improved both models.

Visualising tuning results helps explain why each model behaves differently.

Author

Tom Fitzpatrick
4BCT — University of Galway
