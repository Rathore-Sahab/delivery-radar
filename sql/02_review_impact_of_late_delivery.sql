-- Q2. How much does a late delivery hurt the customer's rating? (test period, May–Aug 2018)
-- Business use: sizes the problem ("the promise is the product").
-- Skills: CASE WHEN, conditional aggregation
SELECT
    CASE WHEN is_late = 1 THEN 'Late' ELSE 'On time' END       AS delivery,
    COUNT(*)                                                     AS orders,
    ROUND(AVG(review_score), 2)                                  AS avg_review,
    ROUND(100.0 * AVG(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END), 1)
                                                                 AS pct_1_or_2_star
FROM 'artifacts/scored_orders.parquet'
WHERE review_score IS NOT NULL
GROUP BY delivery
ORDER BY delivery;
