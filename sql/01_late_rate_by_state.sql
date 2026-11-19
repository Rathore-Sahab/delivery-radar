-- Q1. Which destination states (Brazil) have the worst late-delivery rate? (test period, May–Aug 2018)
-- Business use: where should the ops team focus courier-partner reviews?
-- Skills: GROUP BY, aggregate functions, HAVING, ORDER BY
SELECT
    customer_state                                AS destination_state,
    COUNT(*)                                      AS orders,
    SUM(is_late)                                  AS late_orders,
    ROUND(100.0 * AVG(is_late), 1)                AS late_rate_pct,
    ROUND(AVG(review_score), 2)                   AS avg_review
FROM 'artifacts/scored_orders.parquet'
GROUP BY customer_state
HAVING COUNT(*) >= 200                 -- ignore states with too few orders to judge
ORDER BY late_rate_pct DESC
LIMIT 10;
