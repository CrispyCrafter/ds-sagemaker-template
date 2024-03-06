import datetime as dt
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureUtils:
    def __init__(self, features_, default_features, labels_, categories_):
        self.features_ = features_
        self.default_features = default_features
        self.labels_ = labels_
        self.categories_ = categories_
        self.columns = features_ + default_features + labels_ + categories_

    def slice(self, dataset):
        """
        Dynamically slices the dataset (numpy array or pandas DataFrame)
        based on the provided lists of columns.
        """
        if isinstance(dataset, pd.DataFrame):
            # Slicing pandas DataFrame
            features_slice = dataset[self.features_]
            cohort_date_slice = dataset[["cohort_date"]]
            dx_slice = dataset[["dx"]]
            dnu_slice = dataset[["cohort_size"]]
            labels_slice = dataset[self.labels_]
            categories_slice = dataset[self.categories_] if self.categories_ else None

        else:
            # Slicing numpy array (similar to the previous code)
            features_slice = dataset[:, : len(self.features_)]
            cohort_date_slice = dataset[:, len(self.features_)]
            dx_slice = dataset[:, len(self.features_) + 1]
            dnu_slice = dataset[:, len(self.features_) + 2]

            labels_start_index = len(self.features_) + len(self.default_features)
            labels_end_index = labels_start_index + len(self.labels_) - 1
            labels_slice = dataset[:, labels_start_index : labels_end_index + 1]

            categories_slice = None
            if self.categories_:
                categories_start = labels_end_index + 1
                categories_end = categories_start + len(self.categories_) - 1
                categories_slice = dataset[:, categories_start : categories_end + 1]

        return (
            features_slice,
            cohort_date_slice,
            dx_slice,
            dnu_slice,
            labels_slice,
            categories_slice,
        )

    def extract_features(self, arr):
        return arr[:, 0 : len(self.feature_list)]

    def extract_dx(self, arr):
        return arr[:, len(self.feature_list) + 1]

    def predict_transform(self, df):
        arr = self.transform(df)
        return self.extract_features(arr), self.extract_dx(arr)

    def _dampen_cohort_date_horizon(self, df, max_trend_weeks):
        # should be equal to the encoding cohort code max from feature engineering
        training_n_days = self.encoding_variables["cohort_code_max"]
        new_cohort_code = df["norm_cohort_code"].copy().values

        max_value_cohort_code = 1.0 + (max_trend_weeks * 7.0) / training_n_days

        # by assuming smoothness in transition point cohort_code_max we can
        # compute damp_factor and xs
        damp_factor = (
            max_value_cohort_code - 1
        )  # this was determined for smooth transition
        xs = 1.0 + damp_factor * np.log(max_value_cohort_code - 1)  # determine

        horizon_indices = new_cohort_code[new_cohort_code > 1]
        df.loc[
            df["norm_cohort_code"] > 1, "norm_cohort_code"
        ] = max_value_cohort_code - np.exp(-(horizon_indices - xs) / damp_factor)

    def _dampen_calendar_date_horizon(self, df, max_trend_weeks):
        # should be equal to the encoding cohort code max from feature engineering
        training_n_days = self.encoding_variables["calendar_code_max"]
        new_calendar_code = df["norm_calendar_code"].copy().values

        max_value_calendar_code = 1.0 + (max_trend_weeks * 7.0) / training_n_days

        # by assuming smoothness in transition point calendar_code_max we can
        # compute damp_factor and xs
        damp_factor = (
            max_value_calendar_code - 1
        )  # this was determined for smooth transition
        xs = 1.0 + damp_factor * np.log(max_value_calendar_code - 1)  # determine

        horizon_indices = new_calendar_code[new_calendar_code > 1]
        df.loc[
            df["norm_calendar_code"] > 1, "norm_calendar_code"
        ] = max_value_calendar_code - np.exp(-(horizon_indices - xs) / damp_factor)

    def category_counts(self):
        return {name: len(values) for name, values in self.category_codes.items()}


class FeatureEngineeringTransformer(TransformerMixin, BaseEstimator, FeatureUtils):
    """Feature Engineering transform class for ML inputs"""

    def __init__(
        self,
        features_: List[str],
        categories_: List[str],
        labels_: List[str],
        default_features: List[str] = ["cohort_date", "dx", "cohort_size"],
        **kwargs: Dict[str, Any]
    ):
        super().__init__(features_, default_features, labels_, categories_)

        self.kwargs = kwargs
        self.encoding_variables = {}

        self.log_row_count = None
        self.dnu_row_count = 0
        self.dnu_running_mean = 0
        self.dnu_running_variance = 0

        self.spend_row_count = 0
        self.spend_running_mean = 0
        self.spend_running_variance = 0

        self.category_codes = {cat: {} for cat in self.categories_}

    def fit_transform(self, df):
        """Fit-transform function applied to training (input) domain"""
        self.fit(df)
        df = self.transform(df)
        return df

    def fit(self, df):
        feature_methods = [
            ("norm_cohort_code", self._fit_cohort_date),
            ("norm_calendar_code", self._fit_calendar_date),
            ("log_dnu", self._fit_log_dnu),
            ("log_dx", self._fit_log_dx),
            ("log_spend", self._fit_log_spend),
        ]
        for feature, method in feature_methods:
            if self.kwargs["features"].get(feature):
                method(df)

        self._fit_categories(df)

    def transform(self, df, to_numpy=True):
        df = df.pipe(preprocessing_features, self.kwargs["start_input_date"])
        transform_methods = [
            ("norm_cohort_code", self._transform_cohort_date),
            ("norm_calendar_code", self._transform_calendar_date),
            ("log_dnu", self._transform_log_dnu),
            ("log_dx", self._transform_log_dx),
            ("log_spend", self._transform_log_spend),
        ]
        for feature, method in transform_methods:
            if self.kwargs["features"].get(feature):
                method(df)
        self._transform_categories(df)

        tmp = df.assign(
            cohort_date=lambda df: df["cohort_date"].map(dt.date.toordinal)
        )[self.columns].astype(np.float32)

        if to_numpy:
            return tmp.to_numpy()
        return tmp

    def _fit_cohort_date(self, df):
        cohort_min_date = df["cohort_date"].min()
        cohort_code = (df["cohort_date"] - cohort_min_date).dt.days.values
        cohort_code_max = cohort_code.max()

        self.encoding_variables["cohort_min_date"] = min(
            cohort_min_date,
            self.encoding_variables.get("cohort_min_date", cohort_min_date),
        )

        self.encoding_variables["cohort_code_max"] = max(
            cohort_code_max,
            self.encoding_variables.get("cohort_code_max", cohort_code_max),
        )

    def _fit_calendar_date(self, df):
        calendar_min_date = df["calendar_date"].min()
        calendar_code = (df["calendar_date"] - calendar_min_date).dt.days.values
        calendar_code_max = calendar_code.max()

        self.encoding_variables["calendar_min_date"] = min(
            calendar_min_date,
            self.encoding_variables.get("cohort_min_date", calendar_min_date),
        )

        self.encoding_variables["calendar_code_max"] = max(
            calendar_code_max,
            self.encoding_variables.get("cohort_code_max", calendar_code_max),
        )

    def _fit_log_dx(self, df):
        self.log_row_count = self.log_row_count or 0

        log_dx_values = np.log1p(df["dx"])

        # Compute the batch mean and max
        batch_mean = log_dx_values.mean()
        batch_max = log_dx_values.max()

        # Update the global mean
        if "mean_log_dx" in self.encoding_variables:
            old_mean = self.encoding_variables["mean_log_dx"]
            batch_size = len(df)
            new_mean = old_mean + (batch_mean - old_mean) * batch_size / (
                self.log_row_count + batch_size
            )
            self.encoding_variables["mean_log_dx"] = new_mean
        else:
            self.encoding_variables["mean_log_dx"] = batch_mean
            self.log_row_count = len(df)

        self.encoding_variables["max_training_log_dx"] = max(
            self.encoding_variables.get("max_training_log_dx", batch_max), batch_max
        )

        # Update the total number of rows processed
        self.log_row_count += len(df)

    def _fit_log_spend(self, df):
        log_spend_values = np.log1p(df["marketing_spend"])

        # Compute the global mean and standard deviation using Welfords method

        for value in log_spend_values:
            self.spend_row_count += 1
            delta = value - self.spend_running_mean
            self.spend_running_mean += delta / self.spend_row_count
            delta2 = value - self.spend_running_mean
            self.spend_running_variance += delta * delta2

        self.encoding_variables["mean_log_spend"] = self.spend_running_mean
        if self.dnu_row_count > 1:
            self.encoding_variables["std_log_spend"] = (
                self.spend_running_variance / (self.spend_row_count - 1)
            ) ** 0.5
        else:
            self.encoding_variables["std_log_spend"] = 0

    def _fit_log_dnu(self, df):
        log_dnu_values = np.log1p(df["cohort_size"])

        # Compute the global mean and standard deviation using Welfords method

        for value in log_dnu_values:
            self.dnu_row_count += 1
            delta = value - self.dnu_running_mean
            self.dnu_running_mean += delta / self.dnu_row_count
            delta2 = value - self.dnu_running_mean
            self.dnu_running_variance += delta * delta2

        self.encoding_variables["mean_log_dnu"] = self.dnu_running_mean
        if self.dnu_row_count > 1:
            self.encoding_variables["std_log_dnu"] = (
                self.dnu_running_variance / (self.dnu_row_count - 1)
            ) ** 0.5
        else:
            self.encoding_variables["std_log_dnu"] = 0

    def _transform_cohort_date(self, df):
        cohort_code = (
            df["cohort_date"] - self.encoding_variables["cohort_min_date"]
        ).dt.days.values
        df["norm_cohort_code"] = (
            cohort_code / self.encoding_variables["cohort_code_max"]
        )
        if self.kwargs["hyperparameters"]["trend_week"] is not None:
            self._dampen_cohort_date_horizon(
                df, self.kwargs["hyperparameters"]["trend_week"]
            )

    def _transform_calendar_date(self, df):
        calendar_code = (
            df["calendar_date"] - self.encoding_variables["calendar_min_date"]
        ).dt.days.values
        df["norm_calendar_code"] = (
            calendar_code / self.encoding_variables["calendar_code_max"]
        )
        if self.kwargs["hyperparameters"]["trend_week"] is not None:
            self._dampen_calendar_date_horizon(
                df, self.kwargs["hyperparameters"]["trend_week"]
            )

    def _transform_log_spend(self, df):
        df["log_spend"] = (
            np.log1p(df["marketing_spend"]) - self.encoding_variables["mean_log_spend"]
        ) / self.encoding_variables["std_log_spend"]

    def _transform_log_dnu(self, df):
        df["log_dnu"] = (
            np.log1p(df["cohort_size"]) - self.encoding_variables["mean_log_dnu"]
        ) / self.encoding_variables["std_log_dnu"]

    def _transform_log_dx(self, df):
        df["log_dx"] = np.log1p(df["dx"]) - self.encoding_variables["mean_log_dx"]
        df.loc[
            df["log_dx"] > self.encoding_variables["max_training_log_dx"], "log_dx"
        ] = self.encoding_variables["max_training_log_dx"]

    def _fit_categories(self, df):
        for cat in self.categories_:
            if self.kwargs["categories"].get(cat):
                current_max_code = max(self.category_codes[cat].values(), default=-1)
                new_categories = set(df[cat].unique()) - set(
                    self.category_codes[cat].keys()
                )
                for new_cat in new_categories:
                    current_max_code += 1
                    self.category_codes[cat][new_cat] = current_max_code

    def _transform_categories(self, df):
        for cat in self.categories_:
            if self.kwargs["categories"].get(cat):
                df[cat] = (
                    df[cat].map(self.category_codes.get(cat)).fillna(-1).astype(int)
                )


def preprocessing_features(df, start_input_date):
    return df.assign(
        smooth_early_dx=lambda df: np.exp(-(df["dx"] - 1) / 10),
        golden_cohort=lambda df: np.where(
            (df["cohort_date"] - pd.to_datetime(start_input_date)).dt.days < 10,
            1.0,
            0.0,
        ),
        day_of_week_sin=lambda df: np.sin(
            df["calendar_date"].dt.isocalendar().day * (2 * np.pi / 7)
        ),
        day_of_week_cos=lambda df: np.cos(
            df["calendar_date"].dt.isocalendar().day * (2 * np.pi / 7)
        ),
        week_of_year_sin=lambda df: np.sin(
            df["calendar_date"].dt.isocalendar().week * (2 * np.pi / 52.15)
        ),
        week_of_year_cos=lambda df: np.cos(
            df["calendar_date"].dt.isocalendar().week * (2 * np.pi / 52.15)
        ),
    ).astype(
        {
            "smooth_early_dx": np.float32,
            "golden_cohort": np.float32,
            "day_of_week_sin": np.float32,
            "day_of_week_cos": np.float32,
            "week_of_year_sin": np.float32,
            "week_of_year_cos": np.float32,
        }
    )
