# Week 7 — Operational Insights

**Input:** `ncr_ride_bookings_cleaned.csv`  
**Rows used:** 148,767  
**Artifacts:** 
- `week7_peak_heatmap.png`
- `week7_peak_top10.csv`
- `week7_cancellation_heatmap.png`
- `week7_cancel_by_vehicle.png` (if vehicle type available)

## Peak Demand (Weekday × Hour)
Top 10 peaks by absolute rides:
- Monday 18:00 — 1,821 rides
- Saturday 18:00 — 1,780 rides
- Wednesday 18:00 — 1,774 rides
- Tuesday 18:00 — 1,752 rides
- Friday 18:00 — 1,747 rides
- Sunday 18:00 — 1,744 rides
- Thursday 18:00 — 1,670 rides
- Sunday 19:00 — 1,627 rides
- Wednesday 17:00 — 1,598 rides
- Saturday 17:00 — 1,589 rides

## “No Driver Found” — Cancellation Hotspots
Highest NDF rates (top 10 cells by Weekday × Hour):
- Wednesday 02:00 — rate=10.94% (N=192)
- Friday 04:00 — rate=10.29% (N=204)
- Saturday 01:00 — rate=9.69% (N=196)
- Sunday 03:00 — rate=9.31% (N=204)
- Tuesday 01:00 — rate=9.19% (N=185)
- Monday 01:00 — rate=9.13% (N=208)
- Tuesday 23:00 — rate=9.04% (N=376)
- Tuesday 00:00 — rate=8.99% (N=178)
- Sunday 01:00 — rate=8.85% (N=192)
- Tuesday 02:00 — rate=8.82% (N=204)

## Recommendations (Staffing • Fleet Mix • Pricing)
- Add surge guardrails and short incentives 5–8 PM for congestion relief and fulfillment.
- Review Go Sedan coverage or assignment rules (highest NDF rate at 7.1%).
- Pilot micro-surge or upfront driver bonuses in the top cancellation windows.