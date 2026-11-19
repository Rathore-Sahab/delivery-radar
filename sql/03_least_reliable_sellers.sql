-- Q3. Which sellers are late most often (with enough orders to be fair)?
-- Business use: seller nudges / account-manager follow-up list.
-- Skills: GROUP BY, HAVING, multiple aggregates
SELECT
    seller_id,
    ANY_VALUE(seller_state)             AS seller_state,
    COUNT(*)                            AS orders,
    SUM(is_late)                        AS late_orders,
    ROUND(100.0 * AVG(is_late), 1)      AS late_rate_pct
FROM 'artifacts/scored_orders.parquet'
GROUP BY seller_id
HAVING COUNT(*) >= 30
ORDER BY late_rate_pct DESC, orders DESC
LIMIT 10;
