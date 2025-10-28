# Solar Power Prediction
Solar power data science project using various weather features to help predict the expected energy output for a solar farm.

## Dataset
The solar energy dataset being used comes from the PVDAQ Solar Data Bounty Prize. We chose to focus on [site 9068](https://openei.org/wiki/PVDAQ/Sites/SR_CO).
Some weather data was included in this dataset, but additional weather data was acquired through [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) and [NSRDB](https://nsrdb.nrel.gov/data-viewer).

## Stakeholders
- Denver, Colorado energy companies
- Power componanies looking to expand solar farm locations

## Results
We start with some baseline models to compare against. These predict the energy is always the mean of the previous years, or the energy at a given time is the same as previous years. The more involved models are trained on 2022, validated on 2023, and tested on 2024. Linear regression models start with a basic feature set using just the solar zenith angle. Additional linear models are built using a variety of modifications: adding more features, applying cosine to the solar zenith angle, and including interaction terms between features. We also used ensemble models. As these models do not inherently have time, lag terms were added as possible features. The parameters for these ensemble models were tuned using grid search and cross-validation in the training year.
| Model Name | RMSE | $`R^{2}`$ |
| -------- | ------- | ------- |
| Mean Baseline | 1232 | -0.001 |
| Prior Year Baseline | 1197 | 0.054 |
| Linear Regression | 702 | 0.668 |
| Tuned ExtraTrees | 767 | 0.631 |
| Tuned XGBoost | 768 | 0.630 |

## Features
One main feature used is the solar zenith angle, which is a measure of where the sun is in the sky. Working with hourly weather data, other features that seemed helpful in predicitng were clouds, humidity, precipitation, and temperature. Of these, cloud cover provided the most information. Several weather features were also correlated with each other (e.g. higher precipitation leads to higher humidity), and including too many led to diminishing returns on model performance compared to concerns on overfitting or time spent training.

## Cautions
Need to filter data to include only when the sun was out to possibly provide power. Predicting 0 energy during the night artificially inflates model performance by predicting obvious safe values of 0.
Lag terms require previous measurements, so limit how far in the future predictions can be made.
The features DNI, DHI, and GHI in the weather data are all highly correlated with the target variable of power output; however, these are all variations on measurements of solar energy per unit area. Using these features to predict future solar power would require determining them using prior data, but this would be nearly equivalent to predicting the solar energy output directly.
Ensemble methods provide moderate prediction improvement, but lose interpretibility; however, since we have no control over the weather features knowing exactly how much, for example, cloud cover impacts solar power is not as important.

## Future Directions
The PVDAQ Solar Data Bounty Prize includes several other solar farms, and there are also many other datasets that are not included in the bounty data sets. Other weather features, using more years, or including panel tilt are also possible extensions. With more features, spending additional time tuning parameters for the ensemble methods may be necessary.
