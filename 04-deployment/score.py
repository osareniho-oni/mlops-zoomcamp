#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys
import numpy as np

taxi_type = sys.argv[1]  # 'yellow'
year = int(sys.argv[2])  # 2023
month = int(sys.argv[3])  #  2
    

input_file = f'data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
# input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    dicts = df[categorical].to_dict(orient='records')
    
    return dicts


def apply_model(input_file, output_file):

    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    # Load saved model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Feature matrix
    X = dv.transform(dicts)

    y_pred = model.predict(X)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    # df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    # df_result['PULocationID'] = df['PULocationID']
    # df_result['DOLocationID'] = df['DOLocationID']
    # df_result['actual_duration'] = df['duration']
    # df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    return output_file

def run():
    apply_model(input_file=input_file, output_file=output_file)


if __name__ == '__main__':
    run()

