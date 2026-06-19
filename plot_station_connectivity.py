import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


ROOT = os.path.dirname(os.path.abspath(__file__))
ADJ_CSV = os.path.join(ROOT, "AD_matrix_trimmed_common.csv")
LATLON_CSV = os.path.join(ROOT, "ad_station_latlon.csv")
STATION_TO_IDX_JSON = os.path.join(ROOT, "station_to_idx.json")
OUT_DIR = os.path.join(ROOT, "analysis_results")
OUT_PATH = os.path.join(OUT_DIR, "metro_station_connectivity.png")
OUT_LABELED_PATH = os.path.join(OUT_DIR, "metro_station_connectivity_labeled.png")


def load_coordinates(adj_df):
    latlon_raw = pd.read_csv(LATLON_CSV)
    with open(STATION_TO_IDX_JSON, "r", encoding="utf-8") as fp:
        station_to_idx = json.load(fp)

    full_nodes = pd.DataFrame(
        {
            "station_ad": list(station_to_idx.keys()),
            "node_id": list(station_to_idx.values()),
        }
    )
    latlon_df = full_nodes.merge(latlon_raw, on="station_ad", how="left")
    adj_np = adj_df.values

    missing = latlon_df[latlon_df["lat"].isna() | latlon_df["lon"].isna()]
    for _, row in missing.iterrows():
        node_id = int(row["node_id"])
        neighbors = np.flatnonzero(adj_np[node_id] > 0)
        if len(neighbors) == 0:
            neighbors = np.flatnonzero(adj_np[:, node_id] > 0)
        if len(neighbors) == 0:
            raise ValueError(f"No neighbors available to fill station {node_id}")

        neigh_latlon = latlon_df.iloc[neighbors][["lat", "lon"]].dropna().values
        if len(neigh_latlon) == 0:
            raise ValueError(f"Neighbors of station {node_id} also have missing coordinates")

        latlon_df.loc[latlon_df["node_id"] == node_id, "lat"] = neigh_latlon[:, 0].mean()
        latlon_df.loc[latlon_df["node_id"] == node_id, "lon"] = neigh_latlon[:, 1].mean()

    latlon_df = latlon_df.sort_values("node_id").reset_index(drop=True)

    coords = latlon_df[["lat", "lon"]].to_numpy(dtype=float)
    adj_np = adj_df.values

    # Replace severe coordinate outliers with the mean lat/lon of adjacent nodes.
    corrections = []
    for node_id in range(len(coords)):
        neighbors = np.flatnonzero((adj_np[node_id] > 0) | (adj_np[:, node_id] > 0))
        if len(neighbors) == 0:
            continue

        neighbor_coords = coords[neighbors]
        neighbor_coords = neighbor_coords[~np.isnan(neighbor_coords).any(axis=1)]
        if len(neighbor_coords) == 0:
            continue

        neighbor_mean = neighbor_coords.mean(axis=0)
        distance = np.linalg.norm(coords[node_id] - neighbor_mean)

        if distance > 1.0:
            coords[node_id] = neighbor_mean
            corrections.append((node_id, latlon_df.loc[node_id, "station_ad"], float(distance)))

    latlon_df["lat"] = coords[:, 0]
    latlon_df["lon"] = coords[:, 1]
    latlon_df.attrs["corrections"] = corrections
    return latlon_df


def build_segments(adj_np, coords):
    edges = np.argwhere(adj_np > 0)
    segments = []
    for src, dst in edges:
        segments.append([coords[src], coords[dst]])
    return np.asarray(segments)


def plot_graph(adj_df, latlon_df, out_path, labeled=False):
    adj_np = adj_df.values
    coords = latlon_df[["lon", "lat"]].to_numpy()
    segments = build_segments(adj_np, coords)

    out_degree = (adj_np > 0).sum(axis=1)
    in_degree = (adj_np > 0).sum(axis=0)
    degree = out_degree + in_degree

    fig, ax = plt.subplots(figsize=(11, 11))

    lc = LineCollection(segments, colors="0.65", linewidths=0.6, alpha=0.45, zorder=1)
    ax.add_collection(lc)

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=degree,
        s=18,
        cmap="viridis",
        edgecolors="white",
        linewidths=0.2,
        zorder=2,
    )

    if labeled:
        top_nodes = np.argsort(-degree)[:20]
        for idx in top_nodes:
            ax.text(
                coords[idx, 0],
                coords[idx, 1],
                latlon_df.loc[idx, "station_ad"],
                fontsize=7,
                ha="left",
                va="bottom",
                color="black",
                zorder=3,
            )

    ax.set_title("Metro Station Connectivity")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15, linewidth=0.4)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("In-degree + Out-degree")

    pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.03
    pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.03
    ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    adj_df = pd.read_csv(ADJ_CSV, index_col=0)
    latlon_df = load_coordinates(adj_df)

    plot_graph(adj_df, latlon_df, OUT_PATH, labeled=False)
    plot_graph(adj_df, latlon_df, OUT_LABELED_PATH, labeled=True)

    corrections = latlon_df.attrs.get("corrections", [])
    if corrections:
        print("Corrected coordinate outliers using adjacent-node means:")
        for node_id, station_name, distance in corrections:
            print(f"  node_id={node_id:>3} station={station_name} distance={distance:.3f}")

    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {OUT_LABELED_PATH}")


if __name__ == "__main__":
    main()
