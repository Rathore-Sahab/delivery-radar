# PRD — DeliveryRadar: Late-Delivery Early Warning for Hub Operations

**Author:** Pawan Rathod · **Status:** Prototype shipped (v0.1) · **Live demo:** _<add Streamlit link>_

---

## 1. Problem

E-commerce platforms promise customers a delivery date at checkout. When that promise breaks, the customer's experience collapses. Today, hub and issue-resolution teams find late orders **after** they are late: through customer complaints, "where is my order" tickets, or aged-shipment reports.

Evidence from 96k real orders (Olist, Brazil):

| | On time | Late |
|---|---|---|
| Average review | **4.3 ★** | **2.3 ★** |
| Share of 1–2 ★ reviews | 9% | **61%** |

Ratings stay above 4★ even when a delivery takes long, **as long as the promise is kept**. Missing it by 1–3 days drops the average to 3.3★; missing it by 4–7 days drops it to 2.1★. **The product is the promise, not the speed.**

> Personal context: during my internship with Flipkart's Issue Resolution Team (FSC Pune), the work was reactive. We tracked missing, damaged and orphan shipments after they had already become problems. This prototype asks: *which of today's orders will become tomorrow's escalations?*

## 2. Users & Jobs-to-be-done

| User | Job to be done | Today |
|---|---|---|
| **Hub / ops lead** (primary) | "Each morning, tell me which of today's orders need attention so my small team spends its time where it matters." | Manual checks, random sampling, aged-shipment reports |
| **Customer-experience team** | "Let me warn a customer *before* they have to chase us." | Reactive ticket handling |
| **Storefront / promise team** | "Tell me which delivery promise is safe to show at checkout." | Static estimates per region |

## 3. Goals & Non-goals

**Goals (v1)**
- Rank each day's orders by late-delivery risk **at the moment the order is placed**.
- Give a **reason** and a **suggested action** for every flag, because associates won't trust a black box.
- Beat today's rule-of-thumb triage by a wide margin at a realistic team capacity (10–20% of orders).

**Non-goals (v1)**
- Real-time re-scoring from courier scan events (v2).
- Automatically changing the promise shown at checkout (v3, needs an experiment first).
- Root-causing individual delays.

## 4. Solution

1. **Daily risk queue:** every order gets a risk score, a band (High / Medium / Low), a "why flagged" reason and a suggested action.
2. **Order scorer:** shows the risk for a new order given its lane, its promise and the seller's history. It doubles as a "what promise is safe?" tool.
3. **Two signals, both explainable:**
   - **Delivery buffer** = promised days − the lane's *real* historical transit time (seller-state → customer-state, earlier orders only).
   - **Seller reliability** = the seller's past late rate, shrunk toward the average for new sellers.

| Band | Share of orders | Action |
|---|---|---|
| 🔴 High | top 10% | Prioritise pickup & first-mile scan · alert courier · proactive ETA message to customer |
| 🟠 Medium | next 20% | Hub watch-list · check at next checkpoint · escalate on missed milestone |
| 🟢 Low | rest | Standard flow |

## 5. Success metrics

**North-star:** % of late orders the team catches *before* the promised date passes.

| Metric | Type | Prototype result (held-out May–Aug 2018) |
|---|---|---|
| Late orders caught when reviewing top 10% | Primary | **30%** (rule of thumb: 19%, random: 10%) |
| Late orders caught when reviewing top 20% | Primary | **49%** (rule of thumb: 31%, random: 20%) |
| ROC-AUC | Model health | 0.74 |
| Avg review score on late orders | Outcome (needs A/B) | baseline 2.3★ |
| "Where is my order" tickets per 1k orders | Outcome (needs A/B) | not in dataset |
| Precision of High band (false alarms) | Guardrail | ~13% are truly late (3× the 4.4% base rate). Actions on High must be cheap. |

## 6. Key decisions & trade-offs

1. **Simple model over complex model.** A 21-feature gradient-boosted model scored *worse* on unseen months (AUC 0.68 vs 0.74). In mid-2018 the platform shortened its delivery promises, and the complex model had learned patterns from the old regime. The buffer signal compares each promise with reality, so it adapted automatically. Simple also means every flag can be explained.
2. **Time-based evaluation.** We trained on Sep 2016–Apr 2018 and tested on May–Aug 2018. A random split would overstate accuracy and hide the regime shift.
3. **Optimise for capacity, not accuracy.** The ops team can only touch ~10–20% of orders, so we measure "late orders caught within that capacity", not overall accuracy. The base rate is about 4–7%, so a model that predicts "never late" is 95% accurate and useless.
4. **Accept false alarms on purpose.** Only ~1 in 8 High-risk orders is actually late. That is why the High-band actions are cheap and harmless: a priority scan, or a proactive message.

## 7. Risks & open questions

- **Data recency:** seller late rates use past outcomes. In production, use only orders whose promised date has already passed.
- **Alert fatigue:** if associates ignore the queue, the metric doesn't move. Track the share of flagged orders that are actioned.
- **Customer messaging can backfire:** a "your order may be late" message for an order that arrives on time could still hurt trust. This must be A/B tested.
- **Generalisation:** the order-level model is Brazilian. I tested the core idea on Indian data (26k Delhivery hub-to-hub trips): lane history again was the strongest signal, catching 62% of severe delays in the riskiest 20% of trips (64% with the full model). Indian order-level promises (pin codes, COD, festive peaks) still need their own data.

## 8. Launch plan

| Phase | What | Exit criteria |
|---|---|---|
| 0 — Shadow mode (2 weeks) | Score orders daily, don't act. Compare flags with actual outcomes. | Capture@20% ≥ 40% on live data |
| 1 — Pilot, 1 hub | Hub team acts on the High band | ≥ 60% of flags actioned; no ops slowdown |
| 2 — A/B test proactive messages | 50/50 split of High-risk orders: proactive ETA message vs none | Review ★ on late orders ↑, WISMO tickets ↓ |
| 3 — Scale + v2 | All hubs; add courier-scan re-scoring; promise-setting at checkout | — |

## 9. India evidence (Delhivery hub lanes)

| Finding | Number |
|---|---|
| Real transit vs map-route ETA (median) | 2.0× |
| Trips within ±20% of map ETA | 6% |
| ETA within ±20%: map → DeliveryRadar | 7% → 60% (median error 39 → 12 min) |
| Severe delays caught in riskiest 20% of trips | 64% |
| Severe-delay rate: trips < 20 km vs > 300 km | 34% vs 2% |

**Implication:** in India the promise breaks most often on short, last-mile legs. That's also the problem my Google Maps "Doorstep" case study tackles.

## 10. v2+ ideas

- Re-score in real time when a shipment misses a scan milestone. (This is the signal I worked with at Flipkart: missing and orphan shipments.)
- **Promise optimisation:** show each customer the shortest promise that keeps late risk under X%.
- Seller-facing nudges: "Your last 10 orders were late on lane SP→BA. Here's why."
