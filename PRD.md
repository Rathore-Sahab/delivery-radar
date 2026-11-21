# DeliveryRadar: late-delivery early warning for hub operations

Pawan Rathod · Status: prototype (v0.1) · Demo: [delivery-radar-pawan.streamlit.app](https://delivery-radar-pawan.streamlit.app/)

## 1. Problem

Marketplaces show a promised delivery date at checkout. When that date is missed, customer satisfaction drops sharply. Hub and customer-support teams usually find out about late orders only after the fact, from complaints, "where is my order" tickets or aged-shipment reports.

From 96k delivered orders in the Olist dataset:

| | On time | Late |
|---|---|---|
| Average review | 4.3 | 2.3 |
| Share of 1–2 star reviews | 9% | 61% |

Reviews stay above 4 for on-time orders even when delivery is slow. Orders that arrive 1–3 days late average 3.3, and 4–7 days late average 2.1. Customers react to the missed promise more than to the delivery time itself.

Target setting: an Indian marketplace with seller-fulfilled and FC-fulfilled orders moving through sort hubs and 3PL partners to the last mile. No public Indian dataset has promised and actual delivery dates, so the prototype is built and evaluated on Olist (Brazil). Section 8 covers what changes for India.

Personal context: on Flipkart's issue resolution team (FSC Pune) the work was reactive. We dealt with missing, damaged and orphan shipments after they had become problems. The question here is which of today's orders are likely to become tomorrow's escalations.

## 2. Users

| User | Need | Current approach |
|---|---|---|
| Hub / ops lead (primary) | Each morning, know which orders need attention so a small team spends its time well | Manual checks, random sampling, aged-shipment reports |
| Customer experience | Warn a customer before they have to chase us | Reactive ticket handling |
| Promise / storefront team | Know which delivery promise is safe to show at checkout | Static estimates by region |

## 3. Goals and non-goals

Goals for v1:
- Rank each day's orders by late-delivery risk using only information available when the order is placed.
- Give a reason and a suggested action with every flag. Associates won't act on a score they can't interpret.
- Clearly beat the current rule-of-thumb triage at a realistic capacity (10–20% of orders).

Not in v1:
- Re-scoring from courier scan events (v2).
- Changing the promise shown at checkout automatically (needs an experiment first).
- Root-cause analysis of individual delays.

## 4. Solution

1. **Daily queue:** every order gets a risk score, a band (High / Medium / Low), the main reason it was flagged and a suggested action.
2. **Order scorer:** risk for a hypothetical order given lane, promise and seller history. It also shows the shortest promise that keeps the order in the Low band.
3. **Two inputs, both explainable:**
   - Delivery buffer: promised days minus the lane's observed transit time (seller state to customer state, earlier orders only).
   - Seller late rate: the seller's past late share, shrunk toward the average for sellers with little history.

| Band | Approx. share of orders | Action |
|---|---|---|
| High | top 10% | Priority first-mile pickup and scan, flag to the 3PL's hub, WhatsApp/SMS ETA update; for COD, a confirmation call |
| Medium | next 20% | Sort hub watch list, check at next scan, escalate to the 3PL if a milestone is missed |
| Low | rest | Standard flow |

## 5. Metrics

Primary metric: share of late orders the team identifies before the promised date passes.

| Metric | Type | Prototype result (May–Aug 2018, held out) |
|---|---|---|
| Late orders caught, reviewing top 10% | Primary | 30% (rule: 19%, random: 10%) |
| Late orders caught, reviewing top 20% | Primary | 49% (rule: 31%, random: 20%) |
| ROC-AUC | Model health | 0.74 |
| Avg review on late orders | Outcome, needs A/B test | baseline 2.3 |
| "Where is my order" tickets per 1k orders | Outcome, needs A/B test | not in dataset |
| Precision of High band | Guardrail | ~13% actually late, about 3× the 4.4% base rate |

## 6. Decisions and trade-offs

1. **Simple model over the larger one.** A 21-feature gradient-boosted model scored lower on the test months (AUC 0.68 vs 0.74). Promised delivery windows shortened in mid-2018 and the larger model had fitted the earlier pattern. The buffer feature compares each promise with observed transit, so it adjusted on its own. It is also easier to explain.
2. **Time-based split.** Train on Sep 2016 to Apr 2018, test on May to Aug 2018. A random split would overstate performance and hide the shift in promises.
3. **Measure against capacity, not accuracy.** The late rate is around 4–7%, so a model that always predicts "on time" is about 95% accurate and of no use. The relevant question is how many late orders fall within the slice the team can review.
4. **Accept false alarms.** Only about 1 in 8 High-band orders is actually late, so High-band actions are chosen to be cheap and harmless (a priority scan, a proactive message).

## 7. Risks and open questions

- **Data recency:** seller late rate uses past outcomes. In production it should only count orders whose promised date has already passed.
- **Alert fatigue:** if associates stop working the queue, nothing improves. Track the share of flags that are actioned.
- **Customer messaging:** telling a customer their order may be late, when it then arrives on time, could reduce trust. This needs an A/B test before rollout.
- **Generalisation:** the data is from Brazil. The method should carry over to Indian networks; the numbers won't (see section 8).
- **Festive peaks:** the model has to hold up when volume spikes and buffers shrink network-wide. Evaluate it separately on a festive window before relying on it there.

## 8. Adapting for an Indian marketplace

| Input | Olist prototype | Indian version |
|---|---|---|
| Lane | Seller state → customer state | Origin → destination pin-code cluster, split by 3PL and surface/air |
| Lane transit | Typical days on past orders | Same, rebuilt from 6–12 months of history covering a festive season and a monsoon |
| Promise | Checkout estimate | Same; usually driven by pin-code serviceability tables |
| Seller reliability | Past late share | Past late share plus dispatch-SLA breaches; FC dispatch performance for FC-fulfilled orders |
| New inputs | None | Festive-sale and monsoon flags, COD vs prepaid, first-mile scan time, address quality, 3PL partner |

Where the Indian network is likely to be stressed:
- **Festive sales (Sep–Nov, Diwali):** volume spikes while promises stay roughly fixed, so buffers shrink. In Olist the late rate went from 4% to 12% in the Black Friday month.
- **Monsoon (Jun–Sep):** regional disruption (Mumbai, Assam, Kerala, coastal Karnataka). Lane-level transit captures this better than national averages.
- **Remote pin codes:** the North East, J&K, Ladakh, Andaman and hill districts, with longer and more variable transit and fewer 3PL options.
- **COD and RTO:** a late COD order is a likely refusal and return to origin. Combine lateness risk with RTO risk in the same queue.

## 9. Rollout

| Phase | Scope | Exit criteria |
|---|---|---|
| 0. Rebuild for India | Pin-code-cluster lanes, seller/FC reliability, calendar and COD flags from the platform's own history | Capture @20% on a held-out recent quarter at least matches the Olist result |
| 1. Shadow mode, 2 weeks | One mother hub, outside the festive period. Score daily without acting; compare flags with outcomes | Capture @20% ≥ 40% on live data |
| 2. Pilot, one hub | Hub team acts on the High band | ≥ 60% of flags actioned, no slowdown in normal processing |
| 3. A/B test on customer messages | High-risk orders split 50/50: proactive WhatsApp/SMS ETA update vs none | Higher ratings on late orders, fewer WISMO contacts |
| 4. Festive run, then scale | Run through a sale event with extra High-band capacity; then all hubs, scan-based re-scoring, promise-setting trials | |

## 10. Later

- Re-score when a shipment misses a scan milestone. This is the signal I worked with most at Flipkart (missing and orphan shipments).
- Promise setting: show each customer the shortest promise that keeps predicted risk under a chosen threshold.
- Seller feedback, e.g. "8 of your last 10 orders to BA were late; typical transit on that lane is 18 days."
