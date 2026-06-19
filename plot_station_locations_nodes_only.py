import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from plot_station_connectivity import ADJ_CSV, OUT_DIR, load_coordinates


OUT_PATH = os.path.join(OUT_DIR, "metro_station_locations_outliers_removed_nodes_only.png")
OUT_LABELED_PATH = os.path.join(
    OUT_DIR,
    "metro_station_locations_outliers_removed_nodes_only_labeled.png",
)
KOREAN_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def configure_fonts():
    for font_path in KOREAN_FONT_CANDIDATES:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    return None


def plot_nodes_only(
    adj_df,
    latlon_df,
    out_path,
    labeled=False,
    label_top_k=20,
    label_fontsize=16.5,
    exclude_node_ids=None,
    figsize=(14.3, 14.3),
):
    adj_np = adj_df.values
    plot_df = latlon_df.copy()
    if exclude_node_ids:
        plot_df = plot_df[~plot_df["node_id"].isin(exclude_node_ids)].copy()
    plot_df = plot_df.reset_index(drop=True)

    coords = plot_df[["lon", "lat"]].to_numpy()

    out_degree = (adj_np > 0).sum(axis=1)
    in_degree = (adj_np > 0).sum(axis=0)
    degree_all = out_degree + in_degree
    degree = degree_all[plot_df["node_id"].to_numpy(dtype=int)]

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=degree,
        s=28,
        cmap="viridis",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95,
        zorder=2,
    )

    if labeled:
        top_nodes = np.argsort(-degree)[:label_top_k]
        for idx in top_nodes:
            ax.text(
                coords[idx, 0],
                coords[idx, 1],
                plot_df.loc[idx, "station_ad"],
                fontsize=label_fontsize,
                ha="left",
                va="bottom",
                color="black",
                zorder=3,
            )

    ax.set_title("Metro Station Locations", fontsize=27)
    ax.set_xlabel("Longitude", fontsize=21)
    ax.set_ylabel("Latitude", fontsize=21)
    ax.tick_params(axis="both", labelsize=16.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15, linewidth=0.4)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.012)
    cbar.set_label("In-degree + Out-degree", fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.03
    pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.03
    ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.08, top=0.93)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot station lat/lon locations without connectivity edges."
    )
    parser.add_argument("--out_path", default=OUT_PATH)
    parser.add_argument("--out_labeled_path", default=OUT_LABELED_PATH)
    parser.add_argument("--label_top_k", type=int, default=20)
    parser.add_argument(
        "--keep_outliers",
        action="store_true",
        help="Keep coordinate outlier nodes instead of excluding them from the plot.",
    )
    parser.add_argument(
        "--label_fontsize",
        type=float,
        default=16.5,
        help="Station-label font size. Existing connectivity plot used 7.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    font_name = configure_fonts()

    adj_df = pd.read_csv(ADJ_CSV, index_col=0)
    latlon_df = load_coordinates(adj_df)
    corrections = latlon_df.attrs.get("corrections", [])
    outlier_node_ids = {node_id for node_id, _, _ in corrections}
    exclude_node_ids = None if args.keep_outliers else outlier_node_ids

    plot_nodes_only(
        adj_df,
        latlon_df,
        args.out_path,
        labeled=False,
        exclude_node_ids=exclude_node_ids,
    )
    plot_nodes_only(
        adj_df,
        latlon_df,
        args.out_labeled_path,
        labeled=True,
        label_top_k=args.label_top_k,
        label_fontsize=args.label_fontsize,
        exclude_node_ids=exclude_node_ids,
    )

    if corrections:
        action = "Kept" if args.keep_outliers else "Excluded"
        print(f"{action} coordinate outlier nodes detected by adjacent-node distance:")
        for node_id, station_name, distance in corrections:
            print(f"  node_id={node_id:>3} station={station_name} distance={distance:.3f}")

    print(f"Saved: {args.out_path}")
    print(f"Saved: {args.out_labeled_path}")
    if font_name:
        print(f"Font: {font_name}")


if __name__ == "__main__":
    main()
