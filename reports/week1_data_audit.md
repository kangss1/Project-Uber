# Week 1 — Data Intake & Quality Audit

- File: `ncr_ride_bookings.csv`  
- Rows by Cols: **150,000 x 26**

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
timestamp                            datetime64[ns]
dtype: object
```

## Missing Values Summary (counts + %)

|                                   |   n_missing |   pct_missing |
|:----------------------------------|------------:|--------------:|
| Incomplete Rides                  |      141000 |            94 |
| Incomplete Rides Reason           |      141000 |            94 |
| Cancelled Rides by Customer       |      139500 |            93 |
| Reason for cancelling by Customer |      139500 |            93 |
| Cancelled Rides by Driver         |      123000 |            82 |
| Driver Cancellation Reason        |      123000 |            82 |
| Driver Ratings                    |       57000 |            38 |
| Customer Rating                   |       57000 |            38 |
| Avg CTAT                          |       48000 |            32 |
| Booking Value                     |       48000 |            32 |
| Ride Distance                     |       48000 |            32 |
| Payment Method                    |       48000 |            32 |
| Avg VTAT                          |       10500 |             7 |
| Date                              |           0 |             0 |
| Time                              |           0 |             0 |
| Booking ID                        |           0 |             0 |
| Booking Status                    |           0 |             0 |
| Customer ID                       |           0 |             0 |
| Vehicle Type                      |           0 |             0 |
| Pickup Location                   |           0 |             0 |
| Drop Location                     |           0 |             0 |

## Numeric Profile (selected + IQR outliers)

|                                   |   non_null |           min |           p25 |        median |          p75 |           max |   n_missing |    lower_fence |    upper_fence |   n_outliers_iqr |
|:----------------------------------|-----------:|--------------:|--------------:|--------------:|-------------:|--------------:|------------:|---------------:|---------------:|-----------------:|
| Date                              |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Time                              |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Booking ID                        |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Booking Status                    |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Customer ID                       |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Vehicle Type                      |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Pickup Location                   |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Drop Location                     |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Avg VTAT                          |     139500 |   2           |   5.3         |   8.3         |  11.3        |   20          |       10500 |   -3.7         |   20.3         |                0 |
| Avg CTAT                          |     102000 |  10           |  21.6         |  28.8         |  36.8        |   45          |       48000 |   -1.2         |   59.6         |                0 |
| Cancelled Rides by Customer       |      10500 |   1           |   1           |   1           |   1          |    1          |      139500 |    1           |    1           |                0 |
| Reason for cancelling by Customer |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Cancelled Rides by Driver         |      27000 |   1           |   1           |   1           |   1          |    1          |      123000 |    1           |    1           |                0 |
| Driver Cancellation Reason        |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Incomplete Rides                  |       9000 |   1           |   1           |   1           |   1          |    1          |      141000 |    1           |    1           |                0 |
| Incomplete Rides Reason           |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| Booking Value                     |     102000 |  50           | 234           | 414           | 689          | 4277          |       48000 | -448.5         | 1371.5         |             3435 |
| Ride Distance                     |     102000 |   1           |  12.46        |  23.72        |  36.82       |   50          |       48000 |  -24.08        |   73.36        |                0 |
| Driver Ratings                    |      93000 |   3           |   4.1         |   4.3         |   4.6        |    5          |       57000 |    3.35        |    5.35        |             5203 |
| Customer Rating                   |      93000 |   3           |   4.2         |   4.5         |   4.8        |    5          |       57000 |    3.3         |    5.7         |             3257 |
| Payment Method                    |          0 | nan           | nan           | nan           | nan          |  nan          |      150000 |  nan           |  nan           |                0 |
| timestamp                         |     150000 |   1.70407e+18 |   1.71193e+18 |   1.71983e+18 |   1.7277e+18 |    1.7356e+18 |           0 |    1.68826e+18 |    1.75137e+18 |                0 |
| Hour                              |     150000 |   0           |  10           |  15           |  18          |   23          |           0 |   -2           |   30           |                0 |

## Categorical Cardinality

|                                   |   raw_n_unique |   normalized_n_unique |
|:----------------------------------|---------------:|----------------------:|
| Customer ID                       |         148788 |                148788 |
| Booking ID                        |         148767 |                148767 |
| Time                              |          62910 |                 62910 |
| Date                              |            365 |                   365 |
| Pickup Location                   |            176 |                   176 |
| Drop Location                     |            176 |                   176 |
| Vehicle Type                      |              7 |                     7 |
| Booking Status                    |              5 |                     5 |
| Reason for cancelling by Customer |              5 |                     5 |
| Payment Method                    |              5 |                     5 |
| Driver Cancellation Reason        |              4 |                     4 |
| Incomplete Rides Reason           |              3 |                     3 |

## Key Integrity Checks

- Exact duplicate rows: **0**
- Unique Booking IDs: **148,767**  
- Possible duplicates by ID: **1,233**
- Date NaT: **0**; Hour NaN: **0**; Timestamp NaT: **0**

## Other Issues
- 1,233 possible duplicate booking rows by 'Booking ID'.