# quant-ml

Machine learning research for trading.

## Quick start

```bash
uv sync
uv run python main.py
```

Full setup guide: [docs/00_installation.md](docs/00_installation.md)

## Daily market data

The data pipeline covers 28 training stocks, NBIS outside training, and 18 context series.
Crypto is excluded. Provider snapshots, cleaned tables and quality reports are written
under `data/` and are excluded from Git.

```bash
uv sync
uv run pytest tests/test_data_pipeline.py
uv run python -m quant_ml.data download --symbols RKLB SPY --no-earnings
uv run python -m quant_ml.data build
```

For the full universe, run `uv run python -m quant_ml.data all`.
The tests use synthetic data and make no market-data requests.
They do not approve downloaded data for training.

Universe settings are in [config/data/universe.json](config/data/universe.json).
Independent historical price/volume references must be filled in
[config/data/reference_points.csv](config/data/reference_points.csv) before
split-era historical prices or volumes can be verified.

See [the data pipeline guide](docs/01_data_pipeline.md) for outputs, quality checks,
SPAC cutoffs, SQ/XYZ handling and earnings-calendar limitations.
