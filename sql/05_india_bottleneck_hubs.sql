-- Q5. India: which Delhivery hubs lose the most truck-hours against the map ETA?
-- Business use: where to add dock capacity or re-plan schedules first.
-- Skills: SELECT, ORDER BY, rounding, string functions
SELECT
    REGEXP_REPLACE(dst, ' \(.*\)', '')     AS destination_hub,
    state,
    inbound_legs                           AS inbound_trips,
    ROUND(median_ratio, 1)                 AS real_vs_map_eta,
    ROUND(100.0 * severe_rate, 0)          AS severe_delay_pct,
    ROUND(excess_hours, 0)                 AS hours_beyond_plan
FROM 'artifacts/india_hubs.csv'
ORDER BY hours_beyond_plan DESC
LIMIT 10;
