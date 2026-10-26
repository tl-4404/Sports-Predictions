"""Chronological preprocessing, independent of optional torch/Statcast imports."""
import numpy as np
import pandas as pd


def chronological_features(features, dates, train_fraction=0.8):
    dates = pd.to_datetime(dates)
    days = np.sort(dates.unique())
    if len(days) < 2:
        raise ValueError("at least two game dates required")
    boundary = days[min(len(days) - 1, max(1, int(len(days) * train_fraction)))]
    train = np.flatnonzero(dates < boundary)
    validation = np.flatnonzero(dates >= boundary)
    medians = features.iloc[train].median().fillna(0.0)
    return features.fillna(medians), medians, train, validation
