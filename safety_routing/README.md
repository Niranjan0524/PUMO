# PUMO Safety-Aware Routing

This module prepares road-level safety attributes for the
Personal Urban Mobility Optimizer (PUMO).

## Current work

- Road-network edge processing
- Streetlight-to-road spatial matching
- Flood-location-to-road spatial matching
- Integration of safety attributes
- Dataset validation and quality reporting

## Dataset

The integrated dataset contains 394,508 road-edge records.

Current attributes:
- edge_id
- u, v, key
- streetlight_count
- streetlight_data_available
- nearby_flood_locations
- flood_data_available

## Scripts

- match_streetlights.py
- match_flood.py
- combine_safety.py
- validate_safety_dataset.py

## Limitations

A missing streetlight match does not prove that a road has
no streetlights. Flood-location proximity is not a complete
flood-risk assessment. Accident data and a documented
safety-risk score are future tasks.
