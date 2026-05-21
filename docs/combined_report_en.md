# Urban Light Pollution Classification — Combined Report

## 1. Executive Summary

This project creates a machine learning solution to classify nighttime light pollution levels in Cundinamarca, Colombia. It uses VIIRS satellite radiance data, enriched with an additional balanced dataset of environmental and socioeconomic indicators, to identify low, medium, and high light pollution zones.

The report summarizes the business problem, the available data, the data preparation process, and the main engineering choices made to produce a reliable classification model. It is designed to be used as a concise but complete project narrative for review or Word report preparation.

## 2. Business Context and Objectives

### Problem Statement

Light pollution is a growing environmental concern in urban and peri-urban areas. Excessive nighttime illumination can:
- disturb wildlife and ecosystems
- disrupt human sleep and circadian cycles
- increase energy consumption and waste
- reduce visibility of the night sky

In Cundinamarca, rapid urban expansion around Bogotá, Soacha, and Zipaquirá causes spatial differences in nighttime radiance that are hard to analyze manually. A data-driven classification model can help public authorities identify critical areas and prioritize mitigation efforts.

### Motivation and Value

This project is motivated by the need for:
- spatially explicit evidence of light pollution levels
- reproducible and transparent environmental monitoring
- support for energy and lighting policy decisions
- a public interface for stakeholders to explore results

Deliverables include a cleaned dataset, a trained classification model, and a Flask web application for visualization and navigation through CRISP-ML phases.

### Objectives

The main objectives are:
1. Build a classification model that labels geographic observations in Cundinamarca as Low, Medium, or High light pollution.
2. Prepare a robust dataset with at least 8,000 point observations, including coordinates and radiance values.
3. Explore data quality, spatial distribution, and feature relationships to support model development.
4. Deploy results in a Flask web interface with CRISP-ML phase documentation.

### Stakeholders

- Primary: environmental agencies, regional planners, territorial management teams.
- Secondary: academic researchers in urban sustainability and remote sensing.
- Tertiary: policymakers and community organizations.

## 3. Data Overview

### Primary dataset: VIIRS Nighttime Lights

The core dataset is derived from VIIRS (Visible Infrared Imaging Radiometer Suite) nighttime radiance data. Key characteristics:
- Source: NOAA / NASA public satellite data
- Metric: average radiance (`avg_rad`) in nanoWatts/cm²/sr
- Format: point-level CSV with geographic coordinates
- Observations: 8,360 points covering Cundinamarca
- Spatial resolution: ~750m per pixel

### Supplemental balanced dataset

A second input file, `balanced_dataSet_more_pollutants.csv`, provides polygon-based records with:
- NTL statistics (`ln_mean_NTL`, `ln_sum_NTL`)
- air pollution proxies (`mean_PM25`, `mean_NO2`)
- socioeconomic indicator (`ln_GDPpc`)

This dataset was converted to point-level data using geometry centroid approximation and a scaled proxy for `avg_rad` so it could be merged with the primary VIIRS observations.

### Data dictionary

Key variables in the prepared dataset:
- `system:index`: unique observation identifier
- `avg_rad`: average nighttime radiance
- `.geo`: GeoJSON point geometry
- `lon`, `lat`: geographic coordinates
- `ln_mean_NTL`, `ln_sum_NTL`: log-transformed nightlight features
- `mean_PM25`, `mean_NO2`: air quality proxies
- `ln_GDPpc`: logged GDP per capita

## 4. Exploratory Data Analysis

### Dataset composition

- Total observations: 8,360
- Geographic coverage: Cundinamarca department, including urban and rural areas
- Coordinates fall within expected latitude and longitude bounds for the region

### Radiance distribution

The primary target variable, `avg_rad`, is right-skewed with a long upper tail. Summary statistics:
- mean: ~12.35 nW/cm²/sr
- median: ~8.42 nW/cm²/sr
- min: 0.12
- max: 127.54

Class distribution used for light pollution labels:
- Low (≤ 5): 34%
- Medium (5–15): 41%
- High (>15): 25%

### Data quality and validation

Validation results:
- No missing values in `avg_rad`, `lon`, or `lat`
- No duplicate `system:index` values
- Geographic coordinates are all valid and within the region boundaries
- Radiance values are within a reasonable range, with valid high values for urban hotspots

### Spatial and correlation insights

- High radiance zones are clustered around Bogotá, Soacha-Madrid, and Zipaquirá.
- Rural and remote areas show low radiance values.
- Derived NTL features (`ln_mean_NTL`, `ln_sum_NTL`) correlate strongly with `avg_rad`.
- Air quality proxies show moderate positive correlation with radiance.

## 5. Data Engineering and Preparation

### Data cleaning

The prepared data pipeline includes:
- completeness checks for essential fields
- coordinate validation for all points
- range validation for radiance values
- outlier review without removal, since high values are real urban observations

### Feature engineering

Constructed features for model training:
- `log_rad = log(1 + avg_rad)` to reduce skewness
- `rad_sq = avg_rad^2` to capture non-linear effects
- `dist_bogota` as Euclidean distance to Bogotá centroid to encode urban influence

Optional features available for extended modeling:
- `ln_mean_NTL`, `ln_sum_NTL`
- `mean_PM25`, `mean_NO2`
- `ln_GDPpc`

### Scaling and transformation

Applied standard scaling to numeric features:
- `avg_rad`
- `log_rad`
- `rad_sq`
- `dist_bogota`
- `lat`
- `lon`

Standardization ensures consistent feature ranges for model training and inference.

### Handling outliers and missing values

- No missing values required imputation in the current prepared dataset.
- Outliers in high radiance values were retained because they represent valid urban hotspots.
- The log transformation helped reduce the influence of extreme radiance values.

### Final dataset

The final dataset used by the model is stored in `data_prepared.csv`. It contains the merged VIIRS data plus transformed polygon-derived observations from the balanced dataset.

## 6. Model readiness and feasibility

### Technical feasibility

- The project uses standard Python tools: pandas, scikit-learn, numpy, and Flask.
- The dataset size is manageable for CPU-based training.
- A multi-class classification model is appropriate to classify light pollution levels.
- The Flask application provides a simple way to review results and navigation through CRISP-ML phases.

### Expected outcomes

- A classification model that can distinguish low, medium, and high pollution zones.
- Spatial analysis that highlights areas requiring intervention.
- A reproducible workflow from raw data to prepared dataset and deployed web interface.

### Limitations

- VIIRS spatial resolution (~750m) limits fine-grained urban detail.
- The dataset is based on annual composites and does not capture seasonal variation.
- The balanced dataset uses proxy conversion for `avg_rad`, which is an approximation.
- True ground-truth light pollution measurements are not available for direct validation.

## 7. Recommendations

For a final Word report, include these sections:
1. Executive summary
2. Business problem and objectives
3. Data sources and quality
4. Key EDA findings
5. Data preparation and feature engineering
6. Model approach and feasibility
7. Conclusions and next steps

Optional additions:
- Visual maps of high/medium/low pollution zones
- A diagram of the data pipeline
- A short note on future improvements, such as adding seasonal data or using GIS centroid libraries

## 8. Next steps

- Validate the prepared dataset distributions before final training.
- Keep the current model pipeline and Flask app in sync with `data_prepared.csv`.
- If desired, extend the model with pollutant and socioeconomic features for richer analysis.
- Use this combined English report as the basis for the Word document.
