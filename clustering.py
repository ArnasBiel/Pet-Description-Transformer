# Clustering for data analysis and model enhancement
# KMeans clustering on linguistic features (without TF-IDF of the pet descriptions)

import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

FILES = ["./petfinder_study1.csv", "./petfinder_study2.csv"]

# Linguistic features used for clustering
SELECTED_FEATURES = [
    "Tone", "affect", "posemo", "negemo", "swear",
    "netspeak", "informal", "cogproc", "insight", "cause",
    "social", "family", "friend", "female", "male",
]

N_CLUSTERS = 5


def load_and_combine():
    """Load both CSVs and align their schemas before concatenating."""
    df1 = pl.read_csv(FILES[0], infer_schema_length=10000)

    # df2 has columns that need explicit type overrides
    schema_overrides = {"Sixltr": pl.Float64, "Dic": pl.Float64}
    df2 = pl.read_csv(FILES[1], infer_schema_length=10000, schema_overrides=schema_overrides)

    # Columns present in df1 but missing from df2
    missing_cols = ["published_date", "pull_date", "duration", "ln_duration"]
    for col in missing_cols:
        df2 = df2.with_columns(pl.lit(None).alias(col))

    # Match column order to df1 so we can concatenate
    df2 = df2.select(df1.columns)
    combined_df = pl.concat([df1, df2])

    print(f"Combined shape: {combined_df.shape}")
    print(combined_df.head())

    missing_data = combined_df.null_count()
    print(f"Missing values:\n{missing_data}")

    return combined_df


def run_clustering(combined_df):
    """Standardise features, run KMeans, and attach cluster labels."""
    df_selected = combined_df.select(SELECTED_FEATURES).to_pandas()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_selected)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    kmeans.fit(scaled_features)

    # Cluster distribution
    cluster_counts = pd.Series(kmeans.labels_).value_counts()
    print(f"Cluster distribution:\n{cluster_counts}")

    # Attach labels back to the polars dataframe
    combined_df = combined_df.with_columns(pl.Series(kmeans.labels_).alias("cluster"))
    print(f"Combined DataFrame with clusters:\n{combined_df.head()}")

    return combined_df, scaled_features, kmeans


def visualise_clusters(scaled_features, kmeans):
    """PCA down to 2D and scatter-plot the clusters."""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled_features)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=kmeans.labels_, cmap="viridis", s=50)
    plt.title("KMeans Clustering (PCA-reduced 2D)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig("cluster_visualisation.png", dpi=150)
    plt.show()
    print("Plot saved to cluster_visualisation.png")


if __name__ == "__main__":
    combined_df = load_and_combine()
    combined_df, scaled_features, kmeans = run_clustering(combined_df)
    visualise_clusters(scaled_features, kmeans)
