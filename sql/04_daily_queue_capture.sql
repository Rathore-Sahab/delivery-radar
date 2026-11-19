-- Q4. If the hub team reviews the riskiest 20% of each day's orders,
--     what share of late orders do they catch?
-- Business use: the north-star metric of DeliveryRadar ("capture @ capacity").
-- Skills: CTE (WITH), window function (PERCENT_RANK ... OVER PARTITION BY)
WITH ranked AS (
    SELECT
        CAST(order_purchase_timestamp AS DATE)                          AS order_date,
        is_late,
        PERCENT_RANK() OVER (PARTITION BY CAST(order_purchase_timestamp AS DATE)
                             ORDER BY risk DESC)                        AS risk_rank
    FROM 'artifacts/scored_orders.parquet'
)
SELECT
    COUNT(*)                                                     AS orders,
    SUM(is_late)                                                 AS late_orders,
    SUM(CASE WHEN risk_rank < 0.20 THEN is_late ELSE 0 END)     AS late_caught_in_top_20pct,
    ROUND(100.0 * SUM(CASE WHEN risk_rank < 0.20 THEN is_late ELSE 0 END)
          / SUM(is_late), 1)                                     AS pct_late_caught
FROM ranked;
