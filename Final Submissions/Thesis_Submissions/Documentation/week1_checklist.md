# Week 1 → Week 2 Checklist
- [ ] Impute event columns (0 / "Not Applicable"); finalize numeric imputation plan (KNN vs median).
- [ ] Decide treatment for extreme values in Ride Distance and Booking Value (cap vs remove) — consider using IQR bounds.
- [ ] Confirm handling of rating bounds (clip to [0,5]) and document any changes.
- [ ] Verify Booking ID duplicates; deduplicate with a clear rule if needed.
- [ ] Confirm Date/Time parsing choices and timezone (if applicable); use `timestamp` for resampling.
- [ ] Lock environment (requirements.txt or environment.yml) and commit to repo.