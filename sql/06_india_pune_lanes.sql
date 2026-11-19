-- Q6. India: from Pune hubs, which lanes are the riskiest, and are their
--     destination hubs bottlenecks too?
-- Business use: a Pune hub lead's watch-list.
-- Skills: LEFT JOIN, WHERE with LIKE (note the brackets around OR), string functions
SELECT
    REGEXP_REPLACE(l.src, ' \(.*\)', '') || ' → ' ||
    REGEXP_REPLACE(l.dst, ' \(.*\)', '')           AS lane,
    l.legs                                         AS trips,
    ROUND(l.osrm_km, 0)                            AS km,
    ROUND(l.median_ratio, 1)                       AS real_vs_map_eta,
    ROUND(100.0 * l.severe_rate, 0)                AS lane_severe_pct,
    ROUND(100.0 * h.severe_rate, 0)                AS dest_hub_severe_pct   -- NULL = hub too small to rank
FROM 'artifacts/india_lanes.csv'  AS l
LEFT JOIN 'artifacts/india_hubs.csv' AS h
       ON l.dst = h.dst
WHERE (l.src LIKE 'Pune%' OR l.src LIKE 'PNQ%')
  AND l.legs >= 10
ORDER BY lane_severe_pct DESC, trips DESC
LIMIT 10;
