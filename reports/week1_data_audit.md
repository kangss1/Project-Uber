# Week 1 — Data Intake & Quality Audit

- File: `ncr_ride_bookings.csv`  
- Rows × Cols: **150,000 × 25**

## Schema (dtypes before cleaning)

```
Date                                         object
Time                                         object
Booking ID                                   object
Booking Status                               object
Customer ID                                  object
Vehicle Type                                 object
Pickup Location                              object
Drop Location                                object
Avg VTAT                                    float64
Avg CTAT                                    float64
Cancelled Rides by Customer                 float64
Reason for cancelling by Customer            object
Cancelled Rides by Driver                   float64
Driver Cancellation Reason                   object
Incomplete Rides                            float64
Incomplete Rides Reason                      object
Booking Value                               float64
Ride Distance                               float64
Driver Ratings                              float64
Customer Rating                             float64
Payment Method                               object
Date_parsed                          datetime64[ns]
Hour                                          int32
Weekday                                      object
Month                                        object
dtype: object
```

## Missing Values Summary

```
Incomplete Rides Reason              141000
Incomplete Rides                     141000
Cancelled Rides by Customer          139500
Reason for cancelling by Customer    139500
Driver Cancellation Reason           123000
Cancelled Rides by Driver            123000
Customer Rating                       57000
Driver Ratings                        57000
Ride Distance                         48000
Booking Value                         48000
Payment Method                        48000
Avg CTAT                              48000
Avg VTAT                              10500
Time                                      0
Drop Location                             0
Pickup Location                           0
Vehicle Type                              0
Customer ID                               0
Booking Status                            0
Booking ID                                0
Date                                      0
dtype: int64
```

## Numeric Profile (selected)

|                                   |   non_null |   min |    p25 |   median |    p75 |   max |   n_missing |
|:----------------------------------|-----------:|------:|-------:|---------:|-------:|------:|------------:|
| Date                              |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Time                              |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Booking ID                        |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Booking Status                    |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Customer ID                       |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Vehicle Type                      |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Pickup Location                   |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Drop Location                     |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Avg VTAT                          |     139500 |     2 |   5.3  |     8.3  |  11.3  |    20 |       10500 |
| Avg CTAT                          |     102000 |    10 |  21.6  |    28.8  |  36.8  |    45 |       48000 |
| Cancelled Rides by Customer       |      10500 |     1 |   1    |     1    |   1    |     1 |      139500 |
| Reason for cancelling by Customer |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Cancelled Rides by Driver         |      27000 |     1 |   1    |     1    |   1    |     1 |      123000 |
| Driver Cancellation Reason        |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Incomplete Rides                  |       9000 |     1 |   1    |     1    |   1    |     1 |      141000 |
| Incomplete Rides Reason           |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |
| Booking Value                     |     102000 |    50 | 234    |   414    | 689    |  4277 |       48000 |
| Ride Distance                     |     102000 |     1 |  12.46 |    23.72 |  36.82 |    50 |       48000 |
| Driver Ratings                    |      93000 |     3 |   4.1  |     4.3  |   4.6  |     5 |       57000 |
| Customer Rating                   |      93000 |     3 |   4.2  |     4.5  |   4.8  |     5 |       57000 |
| Payment Method                    |          0 |   nan | nan    |   nan    | nan    |   nan |      150000 |

## Categorical Cardinality

|                                   |   n_unique |
|:----------------------------------|-----------:|
| Customer ID                       |     148788 |
| Booking ID                        |     148767 |
| Time                              |      62910 |
| Date                              |        365 |
| Pickup Location                   |        176 |
| Drop Location                     |        176 |
| Vehicle Type                      |          7 |
| Booking Status                    |          5 |
| Reason for cancelling by Customer |          5 |
| Payment Method                    |          5 |
| Driver Cancellation Reason        |          4 |
| Incomplete Rides Reason           |          3 |

## Key Integrity Checks

- Exact duplicate rows: **0**
- Unique Booking IDs: **148,767**  
- Possible duplicates by ID: **1,233**
- Date parsing NaT count: **0**; Hour NaN count: **0**

## Other Issues
- 1,233 possible duplicate booking rows by 'Booking ID'.