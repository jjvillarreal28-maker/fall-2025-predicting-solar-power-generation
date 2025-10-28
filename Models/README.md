# Solar Power Prediction Models

This folder holds different models trained to predict solar power.

## Baselines

The [baseline models](9068_BaselineModels.ipynb) include:
- Energy is always the mean of previous years.
- Energy at a given time will be the same as previous years.

## Linear

The [linear regression](9068_LinearRegressionModels.ipynb) models range from basic linear models to including additional preprocessing and interaction terms. Models include prediction involving:
- Only solar zenith angle
- Solar zenith angle and cloud cover
- Cosine of solar zenith angle
- Solar zenith angle, cloud cover, relative humidity, and all degree 2 interactions between

## Ensemble

Different [forest regressions](9068_ForestModels.ipynb) were all hyperparameter tuned by training on previous years and testing on a future year. Lag terms were also added to allow the models to partially incorporate time. Models here include:
- Standard random forest using only solar zenith angle
- Standard random forest with solar zenith angle and cloud cover
- ExtraTrees regression trained on weather features and energy 3 hours prior 
- XGBoost including ambient temperature

XGBoost was additionally trained further in [XGBRegression.ipynb](XGBRegression.ipynb).
