import polars


def build_feature_columns(dataframe: polars.DataFrame) -> polars.DataFrame:
    dataframe = dataframe.with_columns([
        polars.col("^*_AT$").str.to_datetime().dt.timestamp(),  # Convert all date columns to a timestamp
        polars.col("^TOTAL_.*$").fill_null(value=-1),  # Fill all count columns with -1 if no data is available
        polars.col("CHURNED").cast(polars.Int8).alias("label")  # Convert label to numerical
    ])
    # Convert date times relative to last login
    dataframe = dataframe.with_columns([
        (polars.col("^*_AT$") - polars.col("LAST_VISIT_AT")).suffix("_DIFF")
    ])
    # Drop the LAST_VISIT_AT_DIFF column as its all zero by definition
    dataframe.drop_in_place("LAST_VISIT_AT_DIFF")

    # Fill null in *_AT_DIFF columns with zero
    dataframe = dataframe.with_columns([
        polars.col("^*_AT_DIFF$").fill_null(strategy="zero")
    ])

    # Pick feature columns
    dataframe = dataframe.select([
        polars.col("label"),
        polars.col("^TOTAL_.*$"),  # Easy numerical columns
        polars.col("^*_AT_DIFF$")  # Cleaned time differences
    ])
    return dataframe
