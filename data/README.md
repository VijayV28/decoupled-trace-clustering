# Raw data — not committed

The BPI Challenge 2019 event log is ~695 MB, which exceeds GitHub's file-size limit, so it is not
part of this repository.

Download `BPI_Challenge_2019.xes` from 4TU.ResearchData and place it in this folder:

<https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853/1>

Expected layout:

```
data/
├── BPI_Challenge_2019.xes     <- you download this
└── interim/                   <- created automatically on first run
    ├── bpi2019_events.parquet
    └── bpi2019_cases.parquet
```

The first pipeline run stream-parses the XES and caches it to parquet under `interim/`; every rerun
reuses the cache. Pass `force_reparse=True` to rebuild from the `.xes`.
