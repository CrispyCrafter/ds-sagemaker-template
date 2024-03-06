import io
import os
import tarfile
from pathlib import Path
from typing import Callable
import datetime as dt

import boto3
import duckdb
import pyarrow as pa
import torch
from pydantic import BaseModel
from pytorch_lightning import LightningDataModule
from sklearn import BaseEstimator
from torch.utils.data import DataLoader, Dataset, random_split


def get_bucket_and_key(uri):
    bucket = uri.split("/")[2]
    key = "/".join(uri.split("/")[3:])
    return bucket, key


class S3ModelLoader:
    def __init__(self, model_path, target="model.pth"):
        bucket_name, file_path = get_bucket_and_key(model_path)
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.target = target

    def __enter__(self):
        s3 = boto3.client("s3")
        try:
            # Download the file into memory
            file_obj = s3.get_object(Bucket=self.bucket_name, Key=self.file_path)
            file_content = file_obj["Body"].read()

            # Unzip the tar.gz file in memory
            tar_gz_file = io.BytesIO(file_content)
            with tarfile.open(fileobj=tar_gz_file, mode="r:gz") as tar:
                # Extract the model.pth file from the tar into memory
                model_file = tar.extractfile(self.target)
                return io.BytesIO(model_file.read())
        except Exception as e:
            print("Failed to read file from S3 bucket. ", e)

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleanup if necessary (e.g., close any open resources)
        pass


def lazy_frame_loader(
    path: str,
    load_all: bool = False,
    data_rows: int = int(512e3),
    data_dropout: float = 0,
    random_seed: int = 32,
    lookback_date: dt.date = dt.date(2023, 1, 1),
):
    path = Path(path)
    if path.is_dir():
        path = f"{path}/*.parquet"

    results = duckdb.sql(
        f"""
    select * from read_parquet("{path}")
    WHERE 
        dx != 0
        AND cohort_size != 0
        AND calendar_date >= strptime('{lookback_date}', '%Y-%m-%d')
    USING SAMPLE {int((1-data_dropout)*100)}% (system, {random_seed});
    """
    )

    if load_all:
        return results.df()

    return iter(results.fetch_arrow_reader(data_rows))


class ParamDataset(Dataset):
    def __init__(self, df_arrow: pa.RecordBatch, transformer: BaseEstimator):
        dataset = torch.tensor(
            transformer.transform(df_arrow.to_pandas()), dtype=torch.float32
        )

        self.len_dataset = dataset.shape[0]
        (
            self.features,
            self.cohort_date,
            self.dx,
            self.dnu,
            self.labels,
            self.categories,
        ) = transformer.slice(dataset)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.cohort_date[idx],
            self.dx[idx],
            self.dnu[idx],
            self.labels[idx],
            self.categories[idx].long(),
        )


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        config: BaseModel,
        transformer: BaseEstimator,
        frame_reader: Callable,
        random_seed=32,
        load_all=False,
    ):
        super().__init__()

        self.data_path = data_path
        self.config = config
        self.random_seed = random_seed
        self.load_all = load_all
        self.transformer = transformer
        self.split_train_test()
        self.frame_reader = frame_reader

    def split_train_test(self):
        frames = self.frame_reader(
            self.data_path,
            load_all=False,
            data_rows=self.config.hyperparameters.data_rows,
            data_dropout=self.config.hyperparameters.data_dropout,
            random_seed=self.random_seed,
            lookback_date=self.config.hyperparameters.lookback_date,
        )

        self.train_val_split = [
            random_split(
                ParamDataset(dataset, self.config, self.transformer),
                [
                    1 - self.config.hyperparameters.validation_split,
                    self.config.hyperparameters.validation_split,
                ],
            )
            for dataset in frames
        ]

    def train_dataloader(self):
        return [
            DataLoader(
                dataset[0],
                batch_size=self.config.hyperparameters.batch_size,
                num_workers=min(os.cpu_count(), 8),
                pin_memory=True,
                shuffle=False,
            )
            for dataset in self.train_val_split
        ]

    def val_dataloader(self):
        return [
            DataLoader(
                dataset[1],
                batch_size=self.config.hyperparameters.batch_size,
                num_workers=min(os.cpu_count(), 8),
                pin_memory=True,
                shuffle=False,
            )
            for dataset in self.train_val_split
        ]
