import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set experiment
mlflow.set_experiment("my-experiment1")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        # Log metrics and parameters
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("min_samples_split", rf.min_samples_split)  # Log the actual value


if __name__ == '__main__':
    run_train()
