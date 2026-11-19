"""Plot the 0N110W monthly NOAA SST anomaly for a merge sanity check."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data/processed/model_data_monthly.csv"
LOOKUP_PATH = ROOT / "data/processed/site_lookup.csv"
PLOT_PATH = ROOT / "reports/noaa_baseline/0N110W_ssta.png"


def main():
    sites = pd.read_csv(LOOKUP_PATH)
    site_id = sites.loc[sites["site_code"].eq("0N110W"), "site_id"].item()
    monthly = pd.read_csv(MODEL_PATH, parse_dates=["month_start"])
    buoy = monthly.loc[monthly["site_id"].eq(site_id)].sort_values("month_start")
    if buoy.empty:
        raise ValueError(f"No monthly model data for {site_id}")

    december = buoy.loc[buoy["month_start"].eq(pd.Timestamp("1997-12-01"))]
    if len(december) != 1:
        raise ValueError("Expected exactly one 0N110W row for December 1997")
    spike = december["ssta"].item()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(buoy["month_start"], buoy["ssta"], linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.scatter(december["month_start"], december["ssta"], color="crimson", zorder=3)
    ax.annotate(
        f"Dec 1997: {spike:+.2f} °C",
        (december["month_start"].item(), spike),
        xytext=(-115, -30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    ax.set(title="0N110W buoy SST anomaly vs NOAA 1981–2010", ylabel="SSTA (°C)")
    ax.set_xlabel("Month")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"December 1997 anomaly: {spike:+.2f} °C")
    print(f"Saved {PLOT_PATH}")


if __name__ == "__main__":
    main()
