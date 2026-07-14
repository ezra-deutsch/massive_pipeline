import os
import io
import logging
from datetime import datetime, timedelta
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import polars as pl
from dotenv import load_dotenv

# Load credentials and S3 endpoint variables from the .env file
load_dotenv()

def setup_logging(log_dir="logs"):
    """
    Configures a centralized logging system to print to console and save
    to execution-specific log files inside the logs/ directory.
    """
    os.makedirs(log_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(log_dir, f"sync_execution_{today_str}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, encoding="utf-8")
        ]
    )
    return logging.getLogger("MarketDataSync")

logger = setup_logging()

def get_s3_client():
    """
    Initializes and returns the Boto3 S3 client pointing to the custom endpoint.
    """
    aws_access_key_id = os.environ.get('MASSIVE_S3_ACCESS_KEY')
    aws_secret_access_key = os.environ.get('MASSIVE_S3_SECRET_KEY')

    if not aws_access_key_id or not aws_secret_access_key:
        logger.critical("Missing AWS credentials in your environment variables.")
        raise ValueError("AWS credentials are not configured in the environment.")

    # Establish custom Boto3 session and point it to the flat files endpoint
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    
    return session.client(
        's3',
        endpoint_url='https://files.massive.com',
        region_name='us-east-1',
        config=Config(signature_version='s3v4'),
    )

def enrich_market_data(df: pl.DataFrame, asset_type: str) -> pl.DataFrame:
    """
    Enriches and parses incoming Polars DataFrames prior to Parquet storage.
    Includes custom temporal and string regex parsers for both asset types.
    """
    # 1. Standardize column names (stripping whitespace, lowercasing)
    df = df.rename({col: col.strip().lower() for col in df.columns})
    
    # 2. Add asset identifier column & tracking columns
    df = df.with_columns([
        pl.lit(asset_type).alias("asset_class"),
        pl.lit(datetime.now().isoformat()).alias("processed_at")
    ])

    # 3. Perform specific transformations depending on asset class
    if asset_type == "stocks":
        # Convert nanosecond timestamp to human-readable UTC datetime
        if "window_start" in df.columns:
            df = df.with_columns(
                pl.from_epoch("window_start", time_unit="ns").dt.replace_time_zone("UTC")
            )
        
        # Ensure symbol is uppercase, compute daily spread if columns exist
        if "symbol" in df.columns:
            df = df.with_columns(pl.col("symbol").str.to_uppercase())
        if "high" in df.columns and "low" in df.columns:
            df = df.with_columns((pl.col("high") - pl.col("low")).alias("high_low_spread"))
            
    elif asset_type == "options":
        enrichments = []
        
        # Convert nanosecond timestamp to human-readable UTC datetime
        if "window_start" in df.columns:
            enrichments.append(
                pl.from_epoch("window_start", time_unit="ns").dt.replace_time_zone("UTC")
            )
        
        # Extract individual components from the OPRA standard options ticker sequence
        if "ticker" in df.columns:
            enrichments.extend([
                # Extract ticker (e.g., 'AAPL') - drops the 'O:' prefix and stops at the expiration date
                pl.col("ticker").str.extract(r"^O:([A-Z0-9]+?)[0-9]{6}[CP]", 1).alias("underlying_ticker"),
                
                # Extract Expiration Date (e.g., '260821' -> '2026-08-21') safely by anchoring to the option type and strike
                pl.col("ticker")
                  .str.extract(r"([0-9]{6})[CP][0-9]{8}$", 1)
                  .str.strptime(pl.Date, format="%y%m%d")
                  .alias("expiration_date"),
                  
                # Extract Option Type ('C' or 'P') safely by anchoring to the strike price at the end
                pl.col("ticker").str.extract(r"([CP])[0-9]{8}$", 1).alias("option_type"),
                
                # Extract Strike Price and convert to actual decimal dollar value
                # (e.g., '00320000' -> 320000 -> divide by 1000 to get 320.00)
                (pl.col("ticker").str.extract(r"([0-9]{8})$", 1).cast(pl.Float64) / 1000.0).alias("strike_price")
            ])
            
        if enrichments:
            df = df.with_columns(enrichments)

        # Additional option enrichments (Mid price if Bid/Ask are available)
        if "bid" in df.columns and "ask" in df.columns:
            df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid_price"))

    return df

def sync_market_data(days_to_check=560, base_data_dir="data", log_dir="logs"):
    """
    Downloads stock and option datasets sequentially from S3, streams them
    into Polars, enriches them, and writes partition-nested Parquet files locally.
    """
    try:
        s3 = get_s3_client()
    except ValueError as e:
        logger.critical(f"Aborting synchronization: {e}")
        return

    today = datetime.now()
    
    # Track the outcome of all sync operations for report logging
    execution_summary = {
        "stocks": {"total": 0, "success": 0, "skipped": 0, "failed": 0},
        "options": {"total": 0, "success": 0, "skipped": 0, "failed": 0}
    }
    report_details = []

    # Map asset configuration keys, S3 prefixes, and templates
    assets_config = {
        "stocks": {
            "s3_prefix": "us_stocks_sip/day_aggs_v1",
            "s3_filename_template": "{date_str}.csv.gz"
        },
        "options": {
            "s3_prefix": "us_options_opra/day_aggs_v1",
            "s3_filename_template": "{date_str}.csv.gz"
        }
    }

    for asset_type, config in assets_config.items():
        logger.info(f"Starting synchronization of {asset_type.upper()} market data...")
        
        for i in range(1, days_to_check + 1):
            target_date = today - timedelta(days=i)
            year_str = target_date.strftime("%Y")
            month_str = target_date.strftime("%m")
            date_str = target_date.strftime("%Y-%m-%d")         # YYYY-MM-DD
            date_nodash = target_date.strftime("%Y%m%d")        # YYYYMMDD
            
            # Construct nested local file path matching the required parquet structure:
            # {base_data_dir}/{asset}/year=YYYY/month=MM/daily_YYYYMMDD.parquet
            local_dir_path = os.path.join(
                base_data_dir, 
                asset_type, 
                f"year={year_str}", 
                f"month={month_str}"
            )
            local_file_name = f"daily_{date_nodash}.parquet"
            local_file_path = os.path.join(local_dir_path, local_file_name)
            
            execution_summary[asset_type]["total"] += 1

            # Check if this optimized Parquet file already exists locally
            if os.path.exists(local_file_path):
                logger.info(f"[{asset_type.upper()}] Parquet file already exists locally: {local_file_name}. Skipping S3 query.")
                execution_summary[asset_type]["skipped"] += 1
                report_details.append(f"{date_str} | {asset_type} | Skipped | Local parquet exists.")
                continue

            # Construct remote S3 path
            s3_filename = config["s3_filename_template"].format(date_str=date_str)
            s3_key = f"{config['s3_prefix']}/{year_str}/{month_str}/{s3_filename}"

            try:
                logger.info(f"[{asset_type.upper()}] Downloading and parsing {s3_filename} directly to memory...")
                
                # Fetch raw CSV.gz bytes from S3
                s3_object = s3.get_object(Bucket="flatfiles", Key=s3_key)
                compressed_bytes = s3_object['Body'].read()
                
                # Polars reads compressed CSV bytes in memory natively
                df = pl.read_csv(io.BytesIO(compressed_bytes))
                
                # Apply data enrichment, transformation, and column parsing rules
                enriched_df = enrich_market_data(df, asset_type)

                # Ensure local target directories exist before exporting
                os.makedirs(local_dir_path, exist_ok=True)
                
                # Write to the nested Parquet structure
                enriched_df.write_parquet(local_file_path, compression="snappy")
                
                logger.info(f"[{asset_type.upper()}] Successfully saved to: {local_file_path}")
                execution_summary[asset_type]["success"] += 1
                report_details.append(f"{date_str} | {asset_type} | Success | Converted CSV.gz -> Parquet")

            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger.warning(f"[{asset_type.upper()}] Data not found on S3 for {date_str} (404).")
                    execution_summary[asset_type]["failed"] += 1
                    report_details.append(f"{date_str} | {asset_type} | Not Found | S3 returned 404")
                else:
                    logger.error(f"[{asset_type.upper()}] S3 Client Error for {date_str}: {e}")
                    execution_summary[asset_type]["failed"] += 1
                    report_details.append(f"{date_str} | {asset_type} | Error | S3 Client Error: {e}")
            except Exception as e:
                logger.error(f"[{asset_type.upper()}] Unexpected error processing {date_str}: {e}")
                execution_summary[asset_type]["failed"] += 1
                report_details.append(f"{date_str} | {asset_type} | Error | Exception: {e}")

    write_summary_report(log_dir, base_data_dir, execution_summary, report_details)

def write_summary_report(log_dir, base_data_dir, summary, report_details):
    """
    Generates a structured TXT execution report saved directly to the log folder.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    report_file_name = f"sync_report_{today_str}.txt"
    report_path = os.path.join(log_dir, report_file_name)

    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write("====================================================\n")
            rf.write(f"UNIFIED MARKET DATA SYNC REPORT - {today_str}\n")
            rf.write("====================================================\n")
            rf.write(f"Execution Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            rf.write(f"Output Data Base:    {os.path.abspath(base_data_dir)}\n")
            rf.write(f"Dedicated Log Dir:   {os.path.abspath(log_dir)}\n\n")
            
            for asset in ["stocks", "options"]:
                rf.write(f"{asset.upper()} SYNCHRONIZATION SUMMARY:\n")
                rf.write("-------------------------------------\n")
                rf.write(f"  Total Days Checked:  {summary[asset]['total']}\n")
                rf.write(f"  Files Converted:     {summary[asset]['success']}\n")
                rf.write(f"  Already Synced:      {summary[asset]['skipped']}\n")
                rf.write(f"  Failed / Unresolved: {summary[asset]['failed']}\n\n")
            
            rf.write("DETAILED RUN CHRONOLOGY:\n")
            rf.write("------------------------\n")
            rf.write("Date       | Asset   | Status   | Notes\n")
            rf.write("-----------|---------|----------|----------------------------------\n")
            for detail in sorted(report_details, reverse=True):
                rf.write(f"{detail}\n")
                
        logger.info(f"Sync report successfully compiled at: {report_path}")
    except Exception as e:
        logger.error(f"Failed to compile run report to disk: {e}")

def main():
    logger.info("Initializing combined stocks and options synchronization pipeline...")
    # Clean output separation config: Data goes to data/, execution logs and reports go to logs/
    sync_market_data(days_to_check=560, base_data_dir="data", log_dir="logs")
    logger.info("Market data pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
