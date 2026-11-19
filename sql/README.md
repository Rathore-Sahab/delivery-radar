# SQL analysis

Six SQL queries that answer real operations questions, run directly on the files in `artifacts/` with [DuckDB](https://duckdb.org/). No database setup is needed.

```bash
pip install duckdb
python sql/run_queries.py      # run from the repo root
```

| # | Question | SQL skills |
|---|---|---|
| 1 | Which Brazilian states have the worst late-delivery rate? | `GROUP BY`, `HAVING`, `ORDER BY` |
| 2 | How much does a late delivery hurt the rating? | `CASE WHEN`, conditional aggregation |
| 3 | Which sellers are late most often? | `GROUP BY`, `HAVING`, multiple aggregates |
| 4 | If the team reviews the riskiest 20% of **each day's** orders, how many late orders are caught? | CTE (`WITH`), window function (`PERCENT_RANK() OVER (PARTITION BY …)`) |
| 5 | India: which Delhivery hubs lose the most truck-hours against the map ETA? | `ORDER BY`, string functions |
| 6 | India: which lanes out of Pune are riskiest, and are their destination hubs bottlenecks? | `LEFT JOIN`, `LIKE`, `NULL` handling |

Queries 1–4 use the held-out test period (May–Aug 2018, 25,239 orders), so their numbers differ slightly from the full-dataset figures in the main README.

**Why query 4 gives 38%, not 49%.** The main README ranks all test-period orders together, and the top 20% catch 49% of late deliveries. Query 4 is stricter and more realistic: the team gets a fixed 20% of *each day's* orders. On bad days there are more late orders than capacity, so the capture rate drops to 38%, still about 2× random checking. This is a good example of why the metric's definition matters.

## Results

### [01_late_rate_by_state.sql](01_late_rate_by_state.sql)

**Q1. Which destination states (Brazil) have the worst late-delivery rate? (test period, May–Aug 2018)**

| destination_state   |   orders |   late_orders |   late_rate_pct |   avg_review |
|:--------------------|---------:|--------------:|----------------:|-------------:|
| BA                  |      833 |            73 |             8.8 |         4.1  |
| ES                  |      478 |            35 |             7.3 |         4.2  |
| PE                  |      410 |            26 |             6.3 |         4.29 |
| SP                  |    11734 |           539 |             4.6 |         4.32 |
| DF                  |      548 |            24 |             4.4 |         4.3  |
| RJ                  |     2929 |           122 |             4.2 |         4.23 |
| GO                  |      466 |            16 |             3.4 |         4.17 |
| CE                  |      282 |             9 |             3.2 |         4.32 |
| MG                  |     2801 |            89 |             3.2 |         4.3  |
| PR                  |     1250 |            38 |             3   |         4.33 |

### [02_review_impact_of_late_delivery.sql](02_review_impact_of_late_delivery.sql)

**Q2. How much does a late delivery hurt the customer's rating? (test period, May–Aug 2018)**

| delivery   |   orders |   avg_review |   pct_1_or_2_star |
|:-----------|---------:|-------------:|------------------:|
| Late       |     1082 |         2.81 |              46.8 |
| On time    |    24047 |         4.36 |               8.3 |

### [03_least_reliable_sellers.sql](03_least_reliable_sellers.sql)

**Q3. Which sellers are late most often (with enough orders to be fair)?**

| seller_id   | seller_state   |   orders |   late_orders |   late_rate_pct |
|:------------|:---------------|---------:|--------------:|----------------:|
| beadbee3    | SP             |       63 |            15 |            23.8 |
| 06a2c3af    | MA             |      291 |            51 |            17.5 |
| c31eff83    | SP             |       68 |            11 |            16.2 |
| 8b9d6eec    | RJ             |       47 |             7 |            14.9 |
| 99a54764    | SP             |       38 |             5 |            13.2 |
| ba143b05    | SP             |       77 |            10 |            13   |
| 6edacfd9    | SP             |       34 |             4 |            11.8 |
| f4649062    | SP             |       79 |             9 |            11.4 |
| fe2032da    | SP             |       72 |             8 |            11.1 |
| e9bc59e7    | PR             |       76 |             8 |            10.5 |

### [04_daily_queue_capture.sql](04_daily_queue_capture.sql)

**Q4. If the hub team reviews the riskiest 20% of each day's orders, what share of late orders do they catch?**

|   orders |   late_orders |   late_caught_in_top_20pct |   pct_late_caught |
|---------:|--------------:|---------------------------:|------------------:|
|    25239 |          1104 |                        419 |                38 |

### [05_india_bottleneck_hubs.sql](05_india_bottleneck_hubs.sql)

**Q5. India: which Delhivery hubs lose the most truck-hours against the map ETA?**

| destination_hub       | state         |   inbound_trips |   real_vs_map_eta |   severe_delay_pct |   hours_beyond_plan |
|:----------------------|:--------------|----------------:|------------------:|-------------------:|--------------------:|
| Gurgaon_Bilaspur_HB   | Haryana       |             928 |               1.9 |                  9 |                4546 |
| Bangalore_Nelmngla_H  | Karnataka     |             665 |               1.8 |                  7 |                3247 |
| Kolkata_Dankuni_HB    | West Bengal   |             219 |               2.6 |                 42 |                2302 |
| Bhiwandi_Mankoli_HB   | Maharashtra   |             583 |               2.3 |                 28 |                2105 |
| Hyderabad_Shamshbd_H  | Telangana     |             405 |               2   |                 19 |                1674 |
| Delhi_Airport_H       | Delhi         |             188 |               2.1 |                 17 |                1504 |
| Pune_Tathawde_H       | Maharashtra   |             330 |               2.1 |                 12 |                1236 |
| Muzaffrpur_Bbganj_I   | Bihar         |             158 |               5.6 |                 68 |                1222 |
| Chandigarh_Mehmdpur_H | Punjab        |             431 |               2.1 |                 15 |                1156 |
| Kanpur_Central_H_6    | Uttar Pradesh |             172 |               2.2 |                 24 |                 945 |

### [06_india_pune_lanes.sql](06_india_pune_lanes.sql)

**Q6. India: from Pune hubs, which lanes are the riskiest, and are their destination hubs bottlenecks too?**

| lane                                    |   trips |   km |   real_vs_map_eta |   lane_severe_pct | dest_hub_severe_pct   |
|:----------------------------------------|--------:|-----:|------------------:|------------------:|:----------------------|
| Pune_Tathawde_H → Pune_PC               |      17 |   25 |               2.7 |                41 | NULL                  |
| Pune_PC → Pune_Tathawde_H               |      18 |   25 |               2.7 |                39 | 12.0                  |
| Pune_Tathawde_H → PNQ Vadgaon Sheri DPC |      43 |   26 |               2.6 |                35 | NULL                  |
| Pune_Tathawde_H → PNQ Pashan DPC        |      41 |   11 |               1.9 |                34 | NULL                  |
| PNQ Vadgaon Sheri DPC → Pune_Tathawde_H |      15 |   31 |               2.4 |                33 | 12.0                  |
| PNQ Pashan DPC → Pune_Tathawde_H        |      32 |   13 |               2   |                31 | 12.0                  |
| Pune_Mundhawa_L → Pune_Tathawde_H       |      33 |   28 |               2.3 |                27 | 12.0                  |
| Pune_North_D_3 → PNQ Rahatani DPC       |      33 |   18 |               2.3 |                21 | NULL                  |
| Pune_Tathawde_H → Pune_Mundhawa_L       |      32 |   26 |               1.8 |                13 | NULL                  |
| Pune_Tathawde_H → Bhiwandi_Mankoli_HB   |     107 |  130 |               2   |                 9 | 28.0                  |
