"""Tests for io.schema."""

import pytest

from rawlobanalyzer.io.schema import (
    LOB_ALL_COLUMNS,
    LOB_CORE_COLUMNS,
    LOB_DERIVED_COLUMNS,
    LOB_SCHEMA,
    MBO_COLUMNS,
    MBO_SCHEMA,
    SchemaValidationError,
    validate_lob_schema,
    validate_mbo_schema,
    validate_parquet_metadata,
)


class TestSchemaConstants:
    def test_lob_column_count(self):
        assert len(LOB_ALL_COLUMNS) == 20
        assert len(LOB_CORE_COLUMNS) == 12
        assert len(LOB_DERIVED_COLUMNS) == 8

    def test_mbo_column_count(self):
        assert len(MBO_COLUMNS) == 6

    def test_lob_schema_field_count(self):
        assert len(LOB_SCHEMA) == 20

    def test_mbo_schema_field_count(self):
        assert len(MBO_SCHEMA) == 6

    def test_no_duplicate_columns(self):
        assert len(set(LOB_ALL_COLUMNS)) == len(LOB_ALL_COLUMNS)
        assert len(set(MBO_COLUMNS)) == len(MBO_COLUMNS)


class TestValidateMetadata:
    def test_valid_metadata(self):
        meta = {
            b"source": b"mbo-lob-reconstructor",
            b"schema_version": b"1.0",
            b"symbol": b"NVDA",
        }
        result = validate_parquet_metadata(meta)
        assert result["source"] == "mbo-lob-reconstructor"
        assert result["symbol"] == "NVDA"

    def test_wrong_source(self):
        meta = {b"source": b"unknown-tool"}
        with pytest.raises(SchemaValidationError, match="Expected source"):
            validate_parquet_metadata(meta)

    def test_none_metadata(self):
        with pytest.raises(SchemaValidationError, match="no metadata"):
            validate_parquet_metadata(None)


class TestSchemaValidation:
    def test_lob_valid(self):
        missing = validate_lob_schema(LOB_SCHEMA)
        assert missing == []

    def test_mbo_valid(self):
        missing = validate_mbo_schema(MBO_SCHEMA)
        assert missing == []
