import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# ===============
# 1. Load Dataset
# =================
housing = pd.read_csv("HOUSING_DATA.csv")

# ====================
# 2. Stratified Split
# ====================

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

housing = strat_train_set.copy()

# 3. Separate Features & Label
# ============================

housing_labels = housing["median_house_value"].copy()

housing_features = housing.drop("median_house_value", axis=1)


# 4. Numerical & Categorical Columns
# ===================================

num_attribs = housing_features.drop(
    "ocean_proximity",
    axis=1
).columns.tolist()

cat_attribs = ["ocean_proximity"]


# 5. Preprocessing Pipeline
# =========================

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])


# 6. Transform Data
# ==================
housing_prepared = full_pipeline.fit_transform(housing_features)

print("=" * 60)
print("Prepared Data Shape :", housing_prepared.shape)
print("=" * 60)


# Function to Train & Evaluate
# =============================

def evaluate_model(model, model_name):

    model.fit(housing_prepared, housing_labels)

    predictions = model.predict(housing_prepared)

    rmse = root_mean_squared_error(
        housing_labels,
        predictions
    )

    scores = -cross_val_score(
        model,
        housing_prepared,
        housing_labels,
        scoring="neg_root_mean_squared_error",
        cv=10
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Training RMSE : {rmse:.2f}")

    print("\nCross Validation RMSE")

    print(pd.Series(scores).describe())

    print("=" * 60)



# Linear Regression
# ==================
linear_model = LinearRegression()

evaluate_model(
    linear_model,
    "Linear Regression"
)

# Decision Tree
# ===============

tree_model = DecisionTreeRegressor(
    random_state=42
)

evaluate_model(
    tree_model,
    "Decision Tree"
)

# Random Forest
# ================

forest_model = RandomForestRegressor(
    random_state=42,
    n_estimators=100
)

evaluate_model(
    forest_model,
    "Random Forest"
)

print("\nTraining Completed Successfully.")

