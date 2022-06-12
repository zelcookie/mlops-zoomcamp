import pandas as pd
from datetime import datetime
import os
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

from prefect import flow, task, get_run_logger

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date=None):
    file_template = os.path.realpath(os.path.join(os.path.dirname(__file__), './data/fhv_tripdata_{date}.parquet'))
    date = datetime.strptime(date, '%Y-%m-%d')
    train_date_str = date.replace(month=date.month-2).strftime('%Y-%m')
    val_date_str = date.replace(month=date.month-1).strftime('%Y-%m')
    train_path, val_path = file_template.format(date=train_date_str),  file_template.format(date=val_date_str)
    return train_path, val_path


@flow
def main(date=None):
    if date is None:
        date = datetime.today().strftime('%Y-%m-%d')
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']
    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    lr_path = os.path.realpath(os.path.join(os.path.dirname(__file__), f'./models/model-{date}.bin'))
    dv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), f'./models/dv-{date}.bin'))
    with open(lr_path, 'wb') as f:
        pickle.dump(lr, f)
    with open(dv_path, 'wb') as f:
        pickle.dump(dv, f)  

    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    flow=main,
    name='hw_3_model_training',
    schedule=CronSchedule(cron='0 9 15 * *'),
    tags=['ml'],
    flow_runner = SubprocessFlowRunner()
)

