import os
from datetime import date
from enum import StrEnum, auto

from pydantic import BaseModel, Field


class CategoryConfig(BaseModel):
    country: bool = False
    platform: bool = False
    channel: bool = False
    network: bool = False
    region: bool = False
    status: bool = False
    attribution: bool = False
    geo_country: bool = False

    def __call__(self):
        return [name for name, value in self.dict().items() if value is True]


class FeatureConfig(BaseModel):
    norm_cohort_code: bool = True
    norm_calendar_code: bool = True
    log_spend: bool = False
    log_dnu: bool = False
    log_dx: bool = True
    smooth_early_dx: bool = True
    golden_cohort: bool = False
    day_of_week_sin: bool = False
    day_of_week_cos: bool = False
    week_of_year_sin: bool = False
    week_of_year_cos: bool = False

    country: bool = False
    platform: bool = False
    channel: bool = False
    network: bool = False
    region: bool = False
    status: bool = False
    attribution: bool = False
    geo_country: bool = False

    def __call__(self):
        return [name for name, value in self.dict().items() if value is True]


class HyperparametersConfig(BaseModel):
    # adjusting
    adjusted_nn: bool = False
    adjusting_new_domain: bool = True
    recent_domain_tail_size: int = 7
    temporal_lookback: int = -1

    # training data parameters
    validation_split: float = 0.2
    min_training_dx: int = 1
    min_training_cohort_size: int = 5
    oversample: bool = False
    trend_week: int = 4
    gradient_clip: float = 1

    # neural network hyperparameters
    batch_size: int = 64
    model_name: str = "pbb"
    dropout_coef: float = 0.2
    l1: int = 32
    l2: int = 16
    loss_function: str = "msle_loss"
    dnu_weight: bool = False
    optimizer_name: str = "sgd"
    learning_rate: float = 0.003
    decayRate: float = 0.95
    n_epochs: int = 5
    min_epoch: int = 10
    gradient_clip: float = 1

    data_shuffle: bool = True
    data_rows: int = 2048e3
    data_dropout: float = 0.1
    lookback_date: date = date(2023, 1, 1)


class Pillar(StrEnum):
    retention = auto()


class NNConfig(BaseModel):
    label_name: Pillar = Field(default_factory=Pillar.retention)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    categories: CategoryConfig = Field(default_factory=CategoryConfig)
    hyperparameters: HyperparametersConfig = Field(
        default_factory=HyperparametersConfig
    )


class EnvConfig:
    # Sagemaker
    train_bucket = f's3://{(os.getenv("DS_SAGE_TRAIN_BUCKET"))}'
    sage_instance_type = os.getenv("DS_SAGE_TRAIN_INSTANCE", "local")

    # Weights and Biases
    wandb_key = os.getenv("DS_WANDB_API_KEY")

    # Ramp Model Repo
    gh_access_token = os.getenv("DS_SAGE_GITHUB_ACCESS_TOKEN")
    ds_model_repo = os.getenv("DS_SAGE_MODEL_REPO")
