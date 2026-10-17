# PUMO Paratransit Fare Model & Assumptions

## 1. Context and Motivation
In urban Indian paratransit operations (such as shared auto-rickshaws in Bengaluru), fare economics are a critical driver of user adoption:
- Passengers are willing to share rides and accept minor detours if they receive a substantial discount compared to private solo auto rides.
- Auto drivers accept deviations and multiple stops if their pooled total revenue exceeds what a single meter fare would yield for the same distance.

---

## 2. Regulatory Solo Auto Baseline (Bengaluru Benchmark)
The standard regulatory auto-rickshaw meter fare in Bengaluru serves as our reference baseline:
- **Base Fare ($F_0$)**: ₹30 for the first 2.0 km.
- **Per-Km Rate ($R_{\text{km}}$)**: ₹15 per km beyond 2.0 km.

For a trip of direct network distance $d_{\text{direct}}$ km:
$$F_{\text{solo}}(d) = F_0 + R_{\text{km}} \cdot \max(0, d_{\text{direct}} - 2.0)$$

Example solo fares:
- 3 km trip: $30 + 15 \times 1 = \text{₹}45$
- 6 km trip: $30 + 15 \times 4 = \text{₹}90$
- 10 km trip: $30 + 15 \times 8 = \text{₹}150$

---

## 3. Shared-Auto Fare Formulation
In our flexible shared-auto system, the passenger's fare is discounted according to two factors:
1. **Sharing Level ($\sigma$)**: The presence of co-passengers sharing segments of the route.
2. **Detour Factor ($\tau$)**: The ratio of actual in-vehicle travel time to direct travel time:
   $$\tau = \frac{T_{\text{ride}}}{T_{\text{direct}}} \ge 1.0$$

### Passenger Fare Formula:
$$F_{\text{shared}} = F_{\text{solo}}(d_{\text{direct}}) \times \left[1.0 - \min\left(\delta_{\max}, \beta_{\text{detour}} \cdot (\tau - 1.0) + \beta_{\text{sharing}} \cdot (N_{\text{shared}} - 1)\right)\right]$$

Where:
- $\delta_{\max} = 0.45$: Maximum allowable passenger discount (passengers pay at least 55% of the solo fare).
- $\beta_{\text{detour}} = 0.25$: Compensation discount per unit of detour ratio.
- $\beta_{\text{sharing}} = 0.15$: Group sharing discount per concurrent passenger party.
- $N_{\text{shared}}$: Average concurrent passenger load during the trip ($\ge 1$).

### Minimum Fare Floor:
Regardless of discounts, a minimum nominal fare of ₹20 is enforced:
$$F_r = \max(20.0, F_{\text{shared}})$$

---

## 4. Multi-Passenger Parties
If a request $r$ reserves $q_r > 1$ seats (e.g., party of 2 or 3 passengers travelling together):
- Base passenger pays $100\%$ of the calculated $F_r$.
- Each additional passenger in the same party receives an additional $40\%$ group discount:
  $$F_r^{\text{total}} = F_r \times (1.0 + 0.6 \times (q_r - 1))$$

---

## 5. Driver Revenue and Platform Incentive
- For vehicle $k$, the total trip revenue for a given stage is:
  $$\text{Revenue}_k = \sum_{r \in \text{served}} F_r^{\text{total}}$$
- Because 2 or 3 passengers each paying 60–75% of solo fare generate 120–225% of solo revenue, the driver enjoys a revenue surplus of +20% to +80% over solo trips, despite driving a modest detour.

---

## 6. Documented Assumptions
1. **Information Transparency**: Passengers are quoted their guaranteed maximum fare and estimated arrival window upon assignment.
2. **Deterministic Rates**: Fuel, congestion, and time-of-day surcharges are held constant across baseline comparisons to isolate algorithm efficiency.
3. **Rational Preference**:
   - In time-first routing, passengers prefer lower travel time over discount.
   - In fare-first routing, passengers prefer higher sharing discounts over speed.
   - In balanced routing, the compromise point on the Pareto curve is selected.
