# Alignment of PUMO Paratransit Routing with Empirical Literature

This document details how three key empirical transportation research papers are operationalized inside the PUMO paratransit routing engine:

1. **Paper 1: The Digital Matatu Project** (Williams et al., 2015, *Journal of Transport Geography*)
2. **Paper 2: Modeling Commuters’ Preference Towards Sharing Paratransit Services** (Das, Bhaduri & Velaga, 2023, *Transport Policy*)
3. **Paper 3: Comparative Evaluation of Commuters' Preferences and Expectations for Sharing Auto-Rickshaw** (Das & Mandal, 2021, *Case Studies on Transport Policy*)

---

## 1. Paper 1: Digital Matatu (Williams et al., 2015)

### Contributions Operationalized in PUMO:
- **Modified GTFS Data Model** ([`gtfs_model.py`](file:///c:/Users/manoj/OneDrive/Desktop/pumo/paratransit_routing/src/pumo_paratransit/gtfs_model.py)):
  - Flexible corridors with designated termini and en-route hail points (`AutoStop`, `SharedAutoRoute`, `RouteShape`, `StopTime`).
  - Spatial mapping of shared-auto stops to directed `RoadGraph` nodes (`road_node_id`) enabling seamless transit routing.
  - Automatic validation pipeline verifying topological connectivity, route integrity, and missing graph references.
- **Bengaluru Shared-Auto Corridors** ([`bengaluru_shared_auto_gtfs.json`](file:///c:/Users/manoj/OneDrive/Desktop/pumo/paratransit_routing/data/bengaluru_shared_auto_gtfs.json)):
  - Five primary arterial corridors modeled on actual high-density shared auto flows in Bengaluru (Majestic–Hebbal North Corridor, Majestic–Banashankari South, Indiranagar–Silk Board, Silk Board–Electronic City Expressway, and ORR East Corridor).

---

## 2. Paper 2: Commuter Preferences & Mixed Logit Model (Das et al., 2023)

### Systematic Utility Specification
The utility of an assignment option for an individual commuter is modeled as:
$$V = \beta_{\text{tt}} \cdot T_{\text{in-vehicle}} + \beta_{\text{wt}} \cdot T_{\text{wait}} + \beta_{\text{at}} \cdot T_{\text{access}} + \beta_{\text{fare}} \cdot \text{Fare} + \beta_{\text{crowd}} \cdot \text{Crowd} + \beta_{\text{info}} \cdot \text{Info}$$

Where:
- $\beta_{\text{wt}} / \beta_{\text{tt}} \approx 1.89$: Commuters value waiting time almost twice as heavily as in-vehicle travel time.
- $\beta_{\text{at}} / \beta_{\text{tt}} \approx 1.44$: First-/last-mile access time (walking to pickup) is heavily penalized.
- $\beta_{\text{info}} > 0$: Real-time trip information significantly enhances commuter utility, especially for female commuters and regular users.

### Generalised Cost Formulation (in ₹)
Using Willingness-to-Pay (WTP) ratios ($WTP_k = \frac{\beta_k}{-\beta_{\text{fare}}}$), PUMO's [`PreferenceWeightedAssigner`](file:///c:/Users/manoj/OneDrive/Desktop/pumo/paratransit_routing/src/pumo_paratransit/preference_assigner.py) evaluates candidate insertions with:
$$\text{Generalised Cost} (\text{₹}) = \text{Fare} + \text{VoTT} \cdot T_{\text{tt}} + \text{VoWT} \cdot T_{\text{wt}} + \text{VoAT} \cdot T_{\text{at}} + \text{Cost}_{\text{crowd}} - \Delta_{\text{realtime\_info}}$$

### Commuter Segmentation
| Segment | Travel Time WTP (₹/min) | Wait Time WTP (₹/min) | Access Time WTP (₹/min) | Real-Time Info WTP (₹/trip) |
|---|---|---|---|---|
| **Regular Commuter** | ₹0.91 | ₹2.43 | ₹1.71 | ₹3.43 |
| **Occasional User** | ₹1.50 | ₹2.83 | ₹2.17 | ₹4.00 |
| **Time-Sensitive** | ₹4.00 | ₹8.50 | ₹5.42 | ₹9.33 |
| **Safety-Conscious** | ₹1.67 | ₹3.46 | ₹2.41 | ₹7.11 |

---

## 3. Paper 3: Expectations & Crowding Deterrent (Das & Mandal, 2021)

### Non-Linear Crowding Discomfort Function
Based on empirical findings that discomfort intensifies non-linearly once vehicle occupancy exceeds 75%:
$$D(\text{occ}) = \begin{cases} 0 & \text{if } \text{occ} \le 0.75 \\ \left(\frac{\text{occ} - 0.75}{1.0 - 0.75}\right)^2 & \text{if } \text{occ} > 0.75 \end{cases}$$

This ensures vehicles approaching full physical capacity incur an increasing utility penalty, discouraging over-allocation and matching passenger expectations of comfort and security.

---

## 4. Empirical Benchmark Comparison

Simulation results on medium demand (60 requests, 8 shared autos) comparing baseline dispatch, standard insertion heuristics, and preference-weighted assigners:

| Strategy / Segment | Service Rate (%) | Avg Wait (min) | Avg Ride (min) | Avg Fare (₹) | Detour Ratio | Avg Generalised Cost (₹) |
|---|---|---|---|---|---|---|
| A2: Solo Ride-Hail (NAV) | 31.7% | 5.71 | 19.74 | ₹139.1 | 1.03 | ₹185.23 |
| A3: Sequential Append (NFV) | 35.0% | 7.13 | 21.70 | ₹172.4 | 1.03 | ₹225.54 |
| A4: Standard Greedy (GIH) | 61.7% | 4.36 | 30.72 | ₹162.0 | 1.24 | ₹220.81 |
| **Pref: Regular Commuter** | **63.3%** | **3.94** | 31.35 | ₹164.7 | 1.28 | **₹202.75** |
| **Pref: Time-Sensitive** | **65.0%** | **3.71** | 30.97 | ₹161.0 | 1.30 | ₹317.89 |
| **Pref: Safety-Conscious** | **65.0%** | **4.10** | 30.48 | ₹159.5 | 1.26 | ₹222.20 |
| **Pref: Occasional User** | **63.3%** | **4.12** | 31.37 | ₹166.8 | 1.28 | ₹225.88 |

### Key Observations:
1. **Wait Time Optimization**: `Time-Sensitive` assignment drives average wait time down to **3.71 minutes** (a 48% reduction compared to A3 baseline).
2. **Economic Efficiency**: `Regular Commuter` assignment achieves an average generalised cost of **₹202.75**, significantly outperforming standard greedy heuristics by optimizing fare-to-detour ratios.
3. **Capacity Protection**: `Safety-Conscious` dispatch avoids over-packing vehicles above the 75% comfort threshold while maintaining a **65.0% service rate**.
