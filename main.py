import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# =================
# File Names
# ==============
DATA_FILE = "HOUSING_DATA.csv"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# ============================
# Build Preprocessing Pipeline
# ============================
def build_pipeline(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return pipeline


# ==================
# Train Model
# =================
def train_model():

    print("=" * 60)
    print("Training Started...")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        print(f"ERROR : {DATA_FILE} not found.")
        return

    housing = pd.read_csv(DATA_FILE)

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

    for train_index, _ in split.split(housing, housing["income_cat"]):
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    labels = housing["median_house_value"].copy()

    features = housing.drop("median_house_value", axis=1)

    num_attribs = features.drop(
        "ocean_proximity",
        axis=1
    ).columns.tolist()

    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(
        num_attribs,
        cat_attribs
    )

    prepared_data = pipeline.fit_transform(features)

    model = RandomForestRegressor(
    n_estimators=30,
    random_state=42,
    n_jobs=-1
)

    model.fit(prepared_data, labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("\nModel Saved Successfully.")
    print("Training Completed.")


# =====================================================
# Predict
# =====================================================

def predict():

    print("=" * 60)
    print("Prediction Started...")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        print(f"""
ERROR !!

'{INPUT_FILE}' file not found.

Create an input.csv file containing:

longitude
latitude
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
ocean_proximity

Then run this program again.
""")

        return

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv(INPUT_FILE)

    transformed = pipeline.transform(input_data)

    predictions = model.predict(transformed)

    input_data["Predicted_House_Price"] = predictions

    input_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nPrediction Completed Successfully.")
    print(f"Output Saved -> {OUTPUT_FILE}")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    if not os.path.exists(MODEL_FILE):

        train_model()

    else:

        predict()