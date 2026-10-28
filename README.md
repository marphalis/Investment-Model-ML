# Investment Model ML (USO)

## Data Engineering

### Initial cleanup

First steps for the project were to look at the structure of the data, round the numeric values to two decimal places and check to ensure there weren't any missing values.

### Validation

Checked if data was consistent with assumption high price > low price.

Investigated time elapsed between the timestamps. The value clusters are: (1,4), (8, 18), (33, 44), (56, 68), (81, 89), 109, 114. I will assume the ranges were as follows: consecutive timestamps with some missing values 1-4 hours, intraday gaps 8-18 hours, one day gaps 33-44 hours, weekend gaps 56-68 hours, anything longer would be treated as an unidentified multiple day break. 

### Open/Close columns importance

Another thing inspected were the gaps between close and the next open prices to see if there were a big discrepancy that should be included in the model.

The average percentage of change between these two values was 0.6% which potentially didn't warrant a separate variable. To make sure there weren't outliers among records after a longer time gap the distributions of time elapsed values with shifts higher than 1% and 0.1% between close and next open prices were analyzed.

| Time Gap        | Diff <= 1% (%) | Diff > 1% (%) | Diff > 0.1% (%) |
|:----------------|:---------------:|:-------------:|:---------------:|
| **consecutive** |      92.86      |     93.44     |      84.60      |
| **intraday**    |      5.60       |     4.85      |      12.0      |
| **one_day**     |      0.07       |     0.09      |      0.13       |
| **weekend**     |      1.29       |     1.35      |      2.86       |
| **long_break**  |      0.19       |     0.27      |      0.41       |

With the 0.1% threshold, there was a difference between the distributions. Consecutive records were less included in the sample with a difference over 0.1%. But with the 1% threshold, distributions were virtually identical, so further analysis was performed with the assumption that the close/next open differences weren't significant enough to warrant additional analysis.

After performing a Chi-squared test to check if the samples come from the same distribution, it further confirmed in the case of 1% difference that these are representative of the full sample and time gaps on average do not influence the close/open difference.

### Statistics

