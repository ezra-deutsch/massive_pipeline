import io
import re
from datetime import date, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st

st.set_page_config(
    page_title="Options Dashboard",
    page_icon="📈",
    layout="wide",
)

DATA_FOLDER = Path(__file__).resolve().parent / "data" / "options"
_DATE_RE = re.compile(r"daily_(\d{8})\.parquet$")

# Schema configuration placeholders
# Update alias or dtype as needed, then reopen the dashboard.
SCHEMA_CONFIG = {
    "option_ticker": {"alias": "ticker", "dtype": pl.Utf8},
    "volume": {"alias": "volume", "dtype": pl.Int64},
    "open": {"alias": "open", "dtype": pl.Decimal(scale=2)},
    "close": {"alias": "close", "dtype": pl.Decimal(scale=2)},
    "high": {"alias": "high", "dtype": pl.Decimal(scale=2)},
    "low": {"alias": "low", "dtype": pl.Decimal(scale=2)},
    "window_start": {"alias": "window_start", "dtype": pl.Date},
    "transactions": {"alias": "transactions", "dtype": pl.Int64},
    "asset_class": {"alias": "asset_class", "dtype": pl.Utf8},
    "processed_at": {"alias": "processed_at", "dtype": pl.Date},
    "underlying_ticker": {"alias": "underlying_ticker", "dtype": pl.Utf8},
    "expiration_date": {"alias": "expiration_date", "dtype": pl.Date},
    "option_type": {"alias": "option_type", "dtype": pl.Utf8},
    "strike_price": {"alias": "strike_price", "dtype": pl.Decimal(scale=2)},
}

@st.cache_data
def list_parquet_files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("daily_*.parquet"))


def parse_date_from_filename(path: Path) -> date | None:
    match = _DATE_RE.match(path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    return None

@st.cache_data
def available_file_dates(folder: Path) -> list[tuple[date, Path]]:
    dates = []
    for path in list_parquet_files(folder):
        parsed = parse_date_from_filename(path)
        if parsed is not None:
            dates.append((parsed, path))
    return sorted(dates)

def apply_schema(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {}
    casts = []
    for source_col, config in SCHEMA_CONFIG.items():
        if source_col in df.columns:
            alias = config.get("alias", source_col)
            dtype = config.get("dtype")
            if alias != source_col:
                rename_map[source_col] = alias
            target_col = source_col if alias == source_col else alias
            if dtype is not None:
                if dtype == pl.Date:
                    if df[target_col].dtype == pl.Utf8:
                        casts.append(
                            pl.col(target_col)
                            .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S.%f", strict=False)
                            .dt.date()
                            .alias(alias)
                        )
                    elif df[target_col].dtype == pl.Datetime:
                        casts.append(pl.col(target_col).dt.date().alias(alias))
                    else:
                        casts.append(pl.col(target_col).cast(pl.Date, strict=False).alias(alias))
                else:
                    casts.append(pl.col(target_col).cast(dtype).alias(alias))

    if rename_map:
        df = df.rename(rename_map)
    if casts:
        df = df.with_columns(casts)
    return df


def format_date_columns_for_display(df: pl.DataFrame) -> pl.DataFrame:
    date_columns = [name for name, dtype in df.schema.items() if dtype == pl.Date]
    if not date_columns:
        return df
    formatted = [pl.col(col).dt.strftime("%Y/%m/%d").alias(col) for col in date_columns]
    return df.with_columns(formatted)

@st.cache_data
def load_available_tickers_for_date_range(file_paths: tuple[str, ...]) -> list[str]:
    tickers: set[str] = set()
    for path in file_paths:
        if not path:
            continue
        df = pl.read_parquet(path, columns=["underlying_ticker"]).unique()
        if not df.is_empty():
            tickers.update(df["underlying_ticker"].to_list())
    return sorted(tickers)

@st.cache_data
def load_filtered_data_for_date_range(file_paths: tuple[str, ...], underlying_ticker: str) -> pl.DataFrame:
    if not file_paths or not underlying_ticker:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for path in file_paths:
        df = pl.scan_parquet(path).filter(pl.col("underlying_ticker") == underlying_ticker).collect()
        if not df.is_empty():
            frames.append(df)

    if not frames:
        return pl.DataFrame()

    return apply_schema(pl.concat(frames))

@st.cache_data
def to_csv_bytes(df: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.write_csv(buffer)
    return buffer.getvalue()

@st.cache_data
def to_parquet_bytes(df: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    return buffer.getvalue()


def build_filters(df: pl.DataFrame) -> pl.DataFrame:
    return df


def build_numeric_chart(df: pl.DataFrame, numeric_columns: list[str]) -> alt.Chart | None:
    if not numeric_columns:
        return None

    numeric_column = numeric_columns[0]
    chart_data = df.select(numeric_column).drop_nulls().to_pandas()
    if chart_data.empty:
        return None

    return alt.Chart(chart_data).mark_bar().encode(
        alt.X(f"{numeric_column}:Q", bin=alt.Bin(maxbins=40), title=numeric_column),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[numeric_column, alt.Tooltip("count():Q", title="Count")],
    ).properties(title=f"Distribution of {numeric_column}")


def build_time_series_chart(df: pl.DataFrame, temporal_columns: list[str], numeric_columns: list[str]) -> alt.Chart | None:
    if not temporal_columns or not numeric_columns:
        return None

    temporal_column = temporal_columns[0]
    value_column = numeric_columns[0]

    chart_df = df.select([temporal_column, value_column]).drop_nulls()
    if chart_df.is_empty():
        return None

    chart_data = chart_df.to_pandas()
    if chart_data.empty:
        return None

    return alt.Chart(chart_data).mark_line(point=True).encode(
        x=alt.X(
            f"{temporal_column}:T",
            title=temporal_column,
            axis=alt.Axis(format="%Y/%m/%d"),
        ),
        y=alt.Y(f"{value_column}:Q", title=value_column),
        tooltip=[temporal_column, value_column],
    ).properties(title=f"{value_column} over {temporal_column}")


def build_category_chart(df: pl.DataFrame, text_columns: list[str]) -> alt.Chart | None:
    if not text_columns:
        return None

    category_column = text_columns[0]

    counts = (
        df.group_by(category_column)
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .limit(20)
    )
    if counts.is_empty():
        return None

    data = counts.to_pandas()
    if data.empty:
        return None

    return alt.Chart(data).mark_bar().encode(
        x=alt.X("count:Q", title="Count"),
        y=alt.Y(f"{category_column}:N", sort="-x", title=category_column),
        tooltip=[category_column, "count:Q"],
    ).properties(title=f"Top 20 {category_column} values")


def build_strike_volume_chart(df: pl.DataFrame) -> alt.Chart | None:
    if not {"strike_price", "volume", "option_type"}.issubset(df.columns):
        return None

    grouped = (
        df.group_by(["strike_price", "option_type"]).agg(pl.sum("volume").alias("volume"))
    )
    if grouped.is_empty():
        return None

    chart_data = grouped.to_pandas()
    if chart_data.empty:
        return None

    return alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("strike_price:O", title="Strike Price", axis=alt.Axis(labelAngle=-45, labelFlush=True)),
        y=alt.Y("volume:Q", title="Total Volume"),
        color=alt.Color("option_type:N", title="Option Type"),
        tooltip=["strike_price:Q", "option_type:N", "volume:Q"],
    ).properties(title="Volume by Strike Price and Option Type")


def main() -> None:
    st.title("Monthly Options Dashboard")
    st.write("Load all daily parquet file from `data/options/` and explore it with filters.")

    parquet_files = list_parquet_files(DATA_FOLDER)
    if not parquet_files:
        st.error(f"No parquet files found in {DATA_FOLDER}")
        return

    file_dates = available_file_dates(DATA_FOLDER)
    if not file_dates:
        st.error(f"No dated parquet files found in {DATA_FOLDER}")
        return

    min_date = file_dates[0][0]
    max_date = file_dates[-1][0]
    selected_dates = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, tuple):
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates

    selected_files = [path for dt, path in file_dates if start_date <= dt <= end_date]
    if not selected_files:
        st.warning("No files available for the selected date range.")
        return

    st.sidebar.write(f"**Selected files:** {len(selected_files)}")
    st.sidebar.write(f"**Date range:** {start_date} to {end_date}")

    underlying_tickers = load_available_tickers_for_date_range(tuple(str(path) for path in selected_files))
    if not underlying_tickers:
        st.warning("No underlying tickers found in the selected date range.")
        return

    selected_underlying = st.sidebar.selectbox(
        "Select underlying ticker",
        ["", *underlying_tickers],
    )

    if not selected_underlying:
        st.info("Select an underlying ticker to view sample rows, visualizations, and download options.")
        return

    df = load_filtered_data_for_date_range(tuple(str(path) for path in selected_files), selected_underlying)
    if df.is_empty():
        st.warning("No rows found for the selected underlying ticker in the chosen date range.")
        return

    st.sidebar.write(f"**Selected ticker:** {selected_underlying}")
    st.sidebar.write(f"**Filtered rows:** {df.height}")

    filtered_df = build_filters(df)
    temporal_columns = [name for name, dtype in filtered_df.schema.items() if dtype == pl.Date]
    if temporal_columns:
        filtered_df = filtered_df.sort(temporal_columns)

    st.subheader("Sample rows")
    display_df = format_date_columns_for_display(filtered_df)
    st.dataframe(display_df, use_container_width=True)

    st.subheader("Visualizations")
    if filtered_df.is_empty():
        st.warning("No rows available after filtering, so no charts can be displayed.")
    else:
        numeric_columns = [name for name, dtype in filtered_df.schema.items() if dtype.is_numeric()]
        temporal_columns = [name for name, dtype in filtered_df.schema.items() if dtype.is_temporal()]
        text_columns = [name for name, dtype in filtered_df.schema.items() if dtype == pl.Utf8]

        strike_chart = build_strike_volume_chart(filtered_df)
        hist_chart = build_numeric_chart(filtered_df, numeric_columns)
        ts_chart = build_time_series_chart(filtered_df, temporal_columns, numeric_columns)
        cat_chart = build_category_chart(filtered_df, text_columns)

        if strike_chart:
            st.altair_chart(strike_chart, use_container_width=True)

        if hist_chart or ts_chart or cat_chart:
            cols = st.columns(2)
            if hist_chart:
                cols[0].altair_chart(hist_chart, use_container_width=True)
            if ts_chart:
                cols[1].altair_chart(ts_chart, use_container_width=True)

            if cat_chart:
                st.altair_chart(cat_chart, use_container_width=True)
        elif not strike_chart:
            st.info("No charts available for the selected columns.")

    st.subheader("Download filtered data")
    csv_bytes = to_csv_bytes(filtered_df)
    parquet_bytes = to_parquet_bytes(filtered_df)
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"options_{selected_underlying}_{start_date}_{end_date}.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download Parquet",
        data=parquet_bytes,
        file_name=f"options_{selected_underlying}_{start_date}_{end_date}.parquet",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
