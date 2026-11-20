# Uploaded Binance Spot BTCUSDT, 15-minute candles

candles.csv combines all six user-uploaded monthly ZIP files for May–October 2026.
17,664 rows; 2026-05-01 00:00 UTC through 2026-10-31 23:45 UTC.
No duplicated timestamps, missing candles or invalid OHLC/volume values were found.
Archive timestamps were checked to be divisible by 1000 and converted from microseconds to milliseconds.
Only open time, open, high, low, close, volume were retained. First 100 rows warm up model features.
The six raw archives are retained in the conversation; source SHA-256 hashes are in provenance.json.
Official exchange CHECKSUM files were not supplied. These are user-uploaded Binance data, not Bybit data.
The default research task can use source=csv, data_file=data/candles.csv, symbol=BTCUSDT, interval=15.

The report in reports/ was generated locally from this exact dataset before upload.
None of the 24 configurations generated a validation entry above its fixed confidence threshold.
No trading profitability was demonstrated. Cash's zero loss is not evidence of a profitable trading model.
The algorithm and confidence thresholds were not changed after observing this dataset.
