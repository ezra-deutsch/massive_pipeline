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

DATA_FOLDER = Path(__file__).resolve().parent / "data" / "options" / "year=2026" / "month=07"
_DATE_RE = re.compile(r"daily_(\d{8})\.parquet$")

@st.cache_data
def list_parquet_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("daily_*.parquet"))


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

@st.cache_data
def load_data_for_date_range(file_paths: tuple[str, ...]) -> pl.DataFrame:
    if not file_paths:
        return pl.DataFrame()
    if len(file_paths) == 1:
        return pl.read_parquet(file_paths[0])
    return pl.concat([pl.read_parquet(path) for path in file_paths])

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
    filtered = df

    text_columns = [name for name, dtype in df.schema.items() if dtype == pl.Utf8]
    numeric_columns = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
    temporal_columns = [name for name, dtype in df.schema.items() if dtype.is_temporal()]

    st.sidebar.header("Filters")
    search_column = st.sidebar.selectbox("Search text column", ["", *text_columns])
    if search_column:
        search_term = st.sidebar.text_input("Contains", value="")
        if search_term:
            filtered = filtered.filter(
                pl.col(search_column)
                .str.to_lowercase()
                .str.contains(search_term.lower(), literal=False)
            )

    if numeric_columns:
        numeric_column = st.sidebar.selectbox("Numeric column", ["", *numeric_columns])
        if numeric_column:
            min_value = float(filtered[numeric_column].min())
            max_value = float(filtered[numeric_column].max())
            range_values = st.sidebar.slider(
                "Value range",
                min_value,
                max_value,
                (min_value, max_value),
                step=max((max_value - min_value) / 100.0, 1.0),
            )
            filtered = filtered.filter(
                pl.col(numeric_column).is_between(range_values[0], range_values[1])
            )

    if temporal_columns:
        temporal_column = st.sidebar.selectbox("Date/time column", ["", *temporal_columns])
        if temporal_column:
            start_date, end_date = st.sidebar.date_input(
                "Range",
                value=(filtered[temporal_column].min().date(), filtered[temporal_column].max().date()),
            )
            filtered = filtered.filter(
                pl.col(temporal_column).is_between(
                    pl.Series([start_date]).cast(pl.Date)[0],
                    pl.Series([end_date]).cast(pl.Date)[0],
                )
            )

    if text_columns:
        category_column = st.sidebar.selectbox("Category column", ["", *text_columns])
        if category_column:
            unique_values = filtered[category_column].unique().to_series().to_list()
            if len(unique_values) <= 50:
                selected_values = st.sidebar.multiselect(
                    f"Show {category_column}", unique_values, default=unique_values[:5]
                )
                if selected_values:
                    filtered = filtered.filter(pl.col(category_column).is_in(selected_values))
            else:
                st.sidebar.write(f"{len(unique_values)} unique values. Use the search filter instead.")

    return filtered


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
        x=alt.X(f"{temporal_column}:T", title=temporal_column),
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
    st.write("Load a daily parquet file from `data/options/year=2026/month=07` and explore it with filters.")

    parquet_files = list_parquet_files(DATA_FOLDER)
    if not parquet_files:
        st.error(f"No parquet files found in {DATA_FOLDER}")
        return

    file_name_map = {file.name: file for file in parquet_files}
    selected_file_name = st.sidebar.selectbox(
        "Choose parquet file",
        options=list(file_name_map.keys()),
        index=max(0, len(file_name_map) - 1),
    )

    file_dates = available_file_dates(DATA_FOLDER)
    if not file_dates:
        st.error(f"No parquet files found in {DATA_FOLDER}")
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

    file_path = selected_files[-1]
    st.sidebar.write(f"**Loaded files:** {len(selected_files)}")
    st.sidebar.write(f"**Date range:** {start_date} to {end_date}")

    df = load_data_for_date_range(tuple(str(path) for path in selected_files))
    st.sidebar.write(f"**Rows before ticker filter:** {df.height}")
    st.sidebar.write(f"**Columns:** {df.width}")

    underlying_tickers = sorted(df.select("underlying_ticker").unique().to_series().to_list())
    selected_underlying = st.sidebar.selectbox(
        "Select underlying ticker",
        ["", *underlying_tickers],
    )

    if not selected_underlying:
        st.info("Select an underlying ticker to view sample rows, visualizations, and download options.")
        return

    df = df.filter(pl.col("underlying_ticker") == selected_underlying)
    st.sidebar.write(f"**Selected ticker:** {selected_underlying}")
    st.sidebar.write(f"**Filtered rows:** {df.height}")

    filtered_df = build_filters(df)

    st.subheader("Data summary")
    st.write("#### Schema")
    st.write(df.schema)
    st.write("#### Sample rows")
    st.dataframe(filtered_df.head(100), use_container_width=True)

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
        file_name=f"{selected_file_name.replace('.parquet', '')}_{selected_underlying}.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download Parquet",
        data=parquet_bytes,
        file_name=f"{selected_file_name.replace('.parquet', '')}_{selected_underlying}.parquet",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
