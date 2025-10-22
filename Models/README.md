# Solar Power Prediction Models

This folder holds different models trained to predict solar power.

## Baselines

The baseline models include:
- Eergy is always the mean of previous years.
- Energy at a given time will be the same as previous years.

## Linear

Linear regression ranges for basic linear models to including additional preprocessing and interaction terms. Models include prediction involving:
- Only solar zenith angle
- Solar zenith angle and cloud cover
- Cosine of solar zenith angle
- Solar zenith angle, cloud cover, relative humidity, and all degree 2 interactions between

## Ensemble

Different forest regressions were all hyperparameter tuned by training on previous years and testing on a future year. Lag terms were also added to allow the models to partially incorporate time. Models include:
- Standard random forest using only solar zenith angle
- Standard random forest with solar zenith angle and cloud cover
- ExtraTrees regression trained on weather features and energy 3 hours prior 
- XGBoost including ambient temperature