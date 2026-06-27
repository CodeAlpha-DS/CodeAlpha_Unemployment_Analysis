from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INDIA_FILE = DATA_DIR / "Unemployment in India.csv"
RATE_FILE = DATA_DIR / "Unemployment_Rate_upto_11_2020.csv"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    rename_map = {
        "Estimated Unemployment Rate (%)": "Unemployment_Rate",
        "Estimated Employed": "Estimated_Employed",
        "Estimated Labour Participation Rate (%)": "Labour_Participation_Rate",
        "Region.1": "Zone",
    }
    df = df.rename(columns=rename_map)
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), dayfirst=True, errors="coerce")
    df["Region"] = df["Region"].astype(str).str.strip()
    if "Area" in df.columns:
        df["Area"] = df["Area"].astype(str).str.strip()
    if "Zone" in df.columns:
        df["Zone"] = df["Zone"].astype(str).str.strip()
    return df


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    india_df = clean_columns(pd.read_csv(INDIA_FILE))
    rate_df = clean_columns(pd.read_csv(RATE_FILE))

    if rate_df.columns.duplicated().any():
        rate_df = rate_df.loc[:, ~rate_df.columns.duplicated()]

    print("Unemployment in India:", india_df.shape)
    print("Unemployment Rate upto 11/2020:", rate_df.shape)
    print("\nMissing values (India dataset):")
    print(india_df.isnull().sum())
    return india_df, rate_df


def national_trend(india_df: pd.DataFrame) -> None:
    monthly = (
        india_df.groupby("Date")["Unemployment_Rate"]
        .mean()
        .reset_index()
        .sort_values("Date")
    )

    plt.figure(figsize=(12, 6))
    plt.plot(monthly["Date"], monthly["Unemployment_Rate"], marker="o", linewidth=2)
    plt.axvline(pd.Timestamp("2020-03-25"), color="red", linestyle="--", label="COVID Lockdown (Mar 2020)")
    plt.title("National Average Unemployment Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "national_trend.png", dpi=150)
    plt.close()

    pre_covid = monthly[monthly["Date"] < "2020-03-01"]["Unemployment_Rate"].mean()
    post_covid = monthly[monthly["Date"] >= "2020-04-01"]["Unemployment_Rate"].mean()
    print(f"\nPre-COVID avg unemployment (before Mar 2020): {pre_covid:.2f}%")
    print(f"Post-lockdown avg unemployment (Apr 2020 onward): {post_covid:.2f}%")
    print(f"Increase: {post_covid - pre_covid:.2f} percentage points")


def covid_impact(india_df: pd.DataFrame) -> None:
    india_df = india_df.copy()
    india_df["Period"] = np.where(
        india_df["Date"] < "2020-03-01",
        "Pre-COVID",
        np.where(india_df["Date"] >= "2020-04-01", "Post-Lockdown", "Transition"),
    )

    impact = (
        india_df[india_df["Period"].isin(["Pre-COVID", "Post-Lockdown"])]
        .groupby(["Region", "Period"])["Unemployment_Rate"]
        .mean()
        .unstack()
        .dropna()
    )
    impact["Increase"] = impact["Post-Lockdown"] - impact["Pre-COVID"]
    top_impact = impact.sort_values("Increase", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    top_impact["Increase"].plot(kind="barh", color="coral")
    plt.title("Top 10 States by Unemployment Increase (Post-Lockdown vs Pre-COVID)")
    plt.xlabel("Increase in Unemployment Rate (%)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "covid_impact_by_state.png", dpi=150)
    plt.close()

    top_impact.to_csv(OUTPUT_DIR / "covid_impact_states.csv")


def state_heatmap(india_df: pd.DataFrame) -> None:
    pivot = india_df.pivot_table(
        index="Region",
        columns=india_df["Date"].dt.to_period("M"),
        values="Unemployment_Rate",
        aggfunc="mean",
    )

    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.1)
    plt.title("State-wise Unemployment Rate Heatmap (Monthly)")
    plt.xlabel("Month")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "state_heatmap.png", dpi=150)
    plt.close()


def rural_urban_analysis(india_df: pd.DataFrame) -> None:
    if "Area" not in india_df.columns:
        return

    area_trend = (
        india_df.groupby(["Date", "Area"])["Unemployment_Rate"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    for area in area_trend["Area"].unique():
        subset = area_trend[area_trend["Area"] == area]
        plt.plot(subset["Date"], subset["Unemployment_Rate"], marker="o", label=area)
    plt.axvline(pd.Timestamp("2020-03-25"), color="red", linestyle="--", label="Lockdown")
    plt.title("Rural vs Urban Unemployment Trends")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rural_urban_trend.png", dpi=150)
    plt.close()


def zone_analysis(rate_df: pd.DataFrame) -> None:
    if "Zone" not in rate_df.columns:
        return

    zone_trend = (
        rate_df.groupby(["Date", "Zone"])["Unemployment_Rate"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    for zone in sorted(zone_trend["Zone"].unique()):
        subset = zone_trend[zone_trend["Zone"] == zone]
        plt.plot(subset["Date"], subset["Unemployment_Rate"], marker="o", label=zone)
    plt.axvline(pd.Timestamp("2020-03-25"), color="red", linestyle="--", label="Lockdown")
    plt.title("Zone-wise Unemployment Trends (2020)")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zone_analysis.png", dpi=150)
    plt.close()


def seasonal_patterns(india_df: pd.DataFrame) -> None:
    india_df = india_df.copy()
    india_df["Month"] = india_df["Date"].dt.month_name()
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    seasonal = (
        india_df.groupby("Month")["Unemployment_Rate"]
        .mean()
        .reindex(month_order)
    )

    plt.figure(figsize=(10, 5))
    seasonal.plot(kind="bar", color="steelblue")
    plt.title("Average Unemployment Rate by Calendar Month (Seasonal Pattern)")
    plt.xlabel("Month")
    plt.ylabel("Average Unemployment Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "seasonal_patterns.png", dpi=150)
    plt.close()


def forecast_trend(india_df: pd.DataFrame) -> None:
    monthly = (
        india_df.groupby("Date")["Unemployment_Rate"]
        .mean()
        .reset_index()
        .sort_values("Date")
    )
    monthly["Month_Index"] = np.arange(len(monthly))

    poly = PolynomialFeatures(degree=2)
    X = poly.fit_transform(monthly[["Month_Index"]])
    model = LinearRegression()
    model.fit(X, monthly["Unemployment_Rate"])

    future_index = np.arange(len(monthly), len(monthly) + 3).reshape(-1, 1)
    future_X = poly.transform(future_index)
    forecast = model.predict(future_X)

    plt.figure(figsize=(12, 6))
    plt.plot(monthly["Date"], monthly["Unemployment_Rate"], marker="o", label="Actual")
    last_date = monthly["Date"].iloc[-1]
    future_dates = pd.date_range(last_date, periods=4, freq="ME")[1:]
    plt.plot(future_dates, forecast, marker="s", linestyle="--", color="green", label="Forecast")
    plt.title("Unemployment Trend with Short-term Forecast")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast.png", dpi=150)
    plt.close()


def write_insights() -> None:
    insights = """# Unemployment Analysis — Key Insights

## COVID-19 Impact
- Unemployment rates rose sharply after the March 2020 national lockdown.
- April 2020 shows the most severe spike across multiple states.
- Recovery began gradually from mid-2020 but rates remained elevated vs 2019.

## Seasonal Trends
- Monthly patterns suggest higher volatility during lockdown months.
- Pre-pandemic months (2019) showed relatively stable unemployment.

## Rural vs Urban
- Both rural and urban areas were affected, with urban regions often showing sharper short-term spikes.

## Policy Implications
- Targeted employment programs in high-impact states could accelerate recovery.
- Rural job schemes and MSME support may reduce post-shock unemployment persistence.
- Continued monitoring of labour participation rates is essential alongside unemployment metrics.
"""
    (OUTPUT_DIR / "insights.md").write_text(insights, encoding="utf-8")


def main() -> None:
    print("CodeAlpha Task 2: Unemployment Analysis with Python\n")
    sns.set_theme(style="whitegrid")
    india_df, rate_df = load_datasets()
    national_trend(india_df)
    covid_impact(india_df)
    state_heatmap(india_df)
    rural_urban_analysis(india_df)
    zone_analysis(rate_df)
    seasonal_patterns(india_df)
    forecast_trend(india_df)
    write_insights()
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
