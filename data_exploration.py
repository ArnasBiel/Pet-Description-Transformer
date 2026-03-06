# Data Exploration
# Examines the combined dataset along the 4 V's: Volume, Variety, Velocity, Veracity

import polars as pl


def explore_volume(df):
    """How much data is there?"""
    rows, columns = df.shape
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {columns}")

    # Rough memory estimate based on dtype sizes
    dtype_sizes = {
        "Int64": 8,
        "Float64": 8,
        "Utf8": 50,     # rough average for string columns
        "String": 50,
        "Boolean": 1,
    }

    memory_usage = 0
    for column in df.columns:
        dtype = str(df[column].dtype)
        col_size = dtype_sizes.get(dtype, 0)
        memory_usage += col_size * df[column].len()

    memory_mb = memory_usage / (1024 ** 2)
    print(f"Estimated memory usage: {memory_mb:.2f} MB")


def explore_variety(df):
    """What types of data do we have?"""
    print(f"Data types:\n{df.dtypes}")

    missing_values = df.null_count()
    print(f"Missing values per column:\n{missing_values}")


def explore_velocity(df):
    """How fast is the data generated or updated?"""
    df = df.with_columns([
        pl.col("published_date").str.strptime(pl.Date, "%m/%d/%y"),
        pl.col("pull_date").str.strptime(pl.Date, "%m/%d/%y"),
    ])
    df = df.sort("published_date")

    min_date = df["published_date"].min()
    max_date = df["published_date"].max()
    print(f"Data covers from {min_date} to {max_date}")

    # Average gap between records
    date_diff = df["published_date"].diff().drop_nulls()
    date_diff_days = date_diff.cast(pl.Float64) / (24 * 60 * 60 * 1000)
    avg_gap = date_diff_days.mean()
    print(f"Average time gap between records: {avg_gap:.2f} days")

    return df


def explore_veracity(df):
    """How reliable and accurate is the data?"""

    # Null values
    null_counts = df.null_count()
    print(f"Null value counts per column:\n{null_counts}")

    # Summary stats for numerical columns
    print(f"Summary statistics:\n{df.describe()}")

    # Unique values in key categorical columns
    categorical_columns = ["pet", "gender", "color_code", "size"]
    for col in categorical_columns:
        if col in df.columns:
            print(f"Unique values in '{col}': {df[col].n_unique()}")

    # Data type consistency
    print(f"Data types:\n{df.dtypes}")

    # Check for duplicate rows
    duplicates_df = df.group_by(df.columns).agg(pl.len().alias("count"))
    duplicate_records = duplicates_df.filter(pl.col("count") > 1)
    num_duplicates = duplicate_records.shape[0]

    print(f"Number of duplicate records: {num_duplicates}")
    if num_duplicates > 0:
        print(f"Duplicate records:\n{duplicate_records}")
    else:
        print("No duplicate records found.")


if __name__ == "__main__":
    # Load and combine the datasets (same logic as clustering.py)
    df1 = pl.read_csv("./petfinder_study1.csv", infer_schema_length=10000)

    schema_overrides = {"Sixltr": pl.Float64, "Dic": pl.Float64}
    df2 = pl.read_csv("./petfinder_study2.csv", infer_schema_length=10000, schema_overrides=schema_overrides)

    missing_cols = ["published_date", "pull_date", "duration", "ln_duration"]
    for col in missing_cols:
        df2 = df2.with_columns(pl.lit(None).alias(col))
    df2 = df2.select(df1.columns)

    combined_df = pl.concat([df1, df2])

    print("=" * 60)
    print("VOLUME")
    print("=" * 60)
    explore_volume(combined_df)

    print("\n" + "=" * 60)
    print("VARIETY")
    print("=" * 60)
    explore_variety(combined_df)

    print("\n" + "=" * 60)
    print("VELOCITY")
    print("=" * 60)
    combined_df = explore_velocity(combined_df)

    print("\n" + "=" * 60)
    print("VERACITY")
    print("=" * 60)
    explore_veracity(combined_df)
