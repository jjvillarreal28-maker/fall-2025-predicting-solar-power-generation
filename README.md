# Solar Power Prediction
Solar power data science project using various weather features to help predict the expected energy output for a solar farm.

## Dataset
The solar energy dataset being used comes from the PVDAQ Solar Data Bounty Prize. We chose to focus on [site 9068](https://openei.org/wiki/PVDAQ/Sites/SR_CO).
Some weather data was included in this dataset, but additional weather data was acquired through [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) and [NSRDB](https://nsrdb.nrel.gov/data-viewer).

## Stakeholders
[solar farms]
[solar energy demand]

## Results
Ensemble methods provide moderate prediction improvement, but lose interpretibility.
| Model Name | RMSE |
| -------- | ------- |
| Mean Baseline | 617.75 |
| Prior Year Baseline | 590.57 |
| Linear Regression Model | 356.16 |
| Tuned ExtraTrees | 339.55 |
| Tuned XGBoost Model | 338.72 |

*one of many

## Features
[chosen list of features from EDA]

## Cautions
Need to filter data to include only when the sun was out to possibly provide power. Predicting 0 energy during the night may artificially inflate model performance.

## Future Directions
The PVDAQ Solar Data Bounty Prize includes several other solar farms, and there are also many other datasets that are not included in the bounty data sets. Other weather features, using more years, or including panel tilt are also possible extensions.
