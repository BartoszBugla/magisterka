"""Validate repository CSV uploads against declared DatasetType schemas."""

from __future__ import annotations

import io

import pandas as pd

from application.dataset_types import (
    DatasetType,
    _SCHEMAS,
)
from config.global_config import (
    SENTIMENT_LABELS,
    FILE_LIMIT,
    NON_ASPECT_COLUMNS,
)

# Synthetic aspect column when a map CSV has coordinates but no aspect labels (home/data_table UI).
REPOSITORY_MAP_SYNTHETIC_ASPECT = "__repository_map_display__"


def validate(data: bytes, dataset_type: DatasetType) -> str | None:
    validate_csv_message = is_csv_valid(data)
    if validate_csv_message is not None:
        return validate_csv_message

    validate_in_limit_message = is_in_limit(data)
    if validate_in_limit_message is not None:
        return validate_in_limit_message

    validate_columns_message = validate_columns(data, dataset_type)
    if validate_columns_message is not None:
        return validate_columns_message

    return None


def is_csv_valid(data: bytes) -> str | None:
    if not data:
        return "No file uploaded."

    try:
        pd.read_csv(io.BytesIO(data))
    except pd.errors.ParserError:
        return "Could not read CSV. file format is not supported."

    return None


def is_in_limit(data: bytes) -> str | None:
    if len(data) > FILE_LIMIT:
        return f"File is too large. Maximum size is {FILE_LIMIT / 1024 / 1024} MB."

    return None


def validate_columns(data: bytes, dataset_type: DatasetType) -> str | None:
    df = pd.read_csv(io.BytesIO(data))

    schema = _SCHEMAS[dataset_type]
    schemas_with_aspects = (DatasetType.LABELLED_AI, DatasetType.LABELLED_HUMAN)

    for column in schema.required_columns:
        if column not in df.columns:
            return f"Dataset does not contain the required column: {column}"

    if dataset_type in schemas_with_aspects:
        validate_aspect_columns_message = validate_aspect_columns(df)
        if validate_aspect_columns_message is not None:
            return validate_aspect_columns_message

    return None


def validate_aspect_columns(df: pd.DataFrame) -> str | None:
    aspect_columns = [
        col for col in df.columns if col.lower() not in NON_ASPECT_COLUMNS
    ]

    if not aspect_columns:
        return "No aspect columns found."

    if not df[aspect_columns].isin(SENTIMENT_LABELS).all(axis=None):
        df[aspect_columns].fillna("notmentioned", inplace=True)

    return None
