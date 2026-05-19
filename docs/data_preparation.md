**Data Preparation — Phase 2 (CRISP-ML(Q))**

Overview
--------
This document describes the steps performed to include and prepare the additional dataset `balanced_dataSet_more_pollutants.csv` and how it was integrated with the existing VIIRS points used by the project.

Goal
----
Produce a single point-level CSV compatible with the project's model input format (`system:index,avg_rad,.geo`) so Phase 2 (Data Preparation) can be reproduced and reviewed. The original VIIRS dataset (`data.csv`) contains point measurements with `avg_rad` (VIIRS radiance) and GeoJSON point coordinates. The newly added dataset contains polygon geometries and several derived variables (NTL logs, pollutants, socio-economic indicators). The chosen approach converts polygon rows into point centroids and builds a proxy `avg_rad` value from available NTL features.

Steps performed
---------------

1. Load existing reference data
   - Read `data.csv` (original VIIRS dataset) to extract reference median of `avg_rad`.
   - This median is used later as a scaling benchmark for the new dataset.

2. Geometry → Centroid (Algorithm: Vertex Averaging)
   
   **Problem**: The balanced dataset contains polygon geometries (WKT-like POLYGON 
   or MULTIPOLYGON strings); the model requires point-level data (lon, lat).
   
   **Algorithm: Approximate Centroid by Vertex Averaging**
   
   PSEUDOCODE:
   ```
   function centroid_from_geometry(geom_string):
       matches = findall(regex: "-?\d+\.\d+\s+-?\d+\.\d+", geom_string)
       lons = [float(match[0]) for each match]
       lats = [float(match[1]) for each match]
       centroid_lon = mean(lons)
       centroid_lat = mean(lats)
       return (centroid_lon, centroid_lat)
   ```
   
   **Implementation details**:
   - Regex pattern: `(-?\d+\.\d+)\s+(-?\d+\.\d+)` extracts all longitude-latitude pairs.
   - For each row in the balanced dataset, extract all coordinate pairs from the geometry string.
   - Compute the arithmetic mean of all longitudes and latitudes separately.
   - This approximation is implemented to avoid introducing additional GIS dependencies 
     (e.g., `shapely`, `geopandas`) and is adequate for coarse-grained point-level inputs.
   - Trade-off: Precision vs. simplicity. True centroid computation would use polygon area 
     weighting, but averaging vertices is sufficient for this use case.
   
   **Null handling**: If no valid coordinates are found, (NaN, NaN) is assigned.

3. Constructing an `avg_rad` proxy (Algorithm: Median-Scaled NTL Proxy)
   
   **Problem**: The original model requires `avg_rad` (VIIRS radiance). The balanced 
   dataset does not contain `avg_rad` but includes night-time lights fields (`ln_mean_NTL`, 
   `ln_sum_NTL`) and pollutants (`mean_PM25`).
   
   **Algorithm: Median-Scaled NTL Log-Inversion**
   
   PSEUDOCODE:
   ```
   function build_avg_rad_proxy(balanced_df, ref_median_avg_rad):
       // Step 1: Choose proxy column (priority: ln_mean_NTL > ln_sum_NTL > mean_PM25)
       if "ln_mean_NTL" in columns:
           proxy_log = balanced_df["ln_mean_NTL"]
       else if "ln_sum_NTL" in columns:
           proxy_log = balanced_df["ln_sum_NTL"]
       else:
           proxy_log = balanced_df.get("mean_PM25", zeros)
       
       // Step 2: Invert logarithm to recover raw magnitude
       raw_proxy = exp(proxy_log)
       
       // Step 3: Handle nulls and invalid values
       raw_proxy = fillna(raw_proxy, 0)  // replace NaN with 0
       raw_proxy = replace(raw_proxy, [inf, -inf], NaN)  // remove infinities
       raw_proxy = fillna(raw_proxy, 0)  // fill again after inf removal
       
       // Step 4: Compute scaling factor using medians (robust to outliers)
       median_ref = median(existing_avg_rad)  // from data.csv
       median_new = median(raw_proxy)
       if median_new > 0 and isfinite(median_new):
           scale = median_ref / median_new
       else:
           scale = 1.0
       
       // Step 5: Apply scaling
       avg_rad = raw_proxy * scale
       return avg_rad
   ```
   
   **Implementation details**:
   - **Null handling**: Use `fillna(0)` to replace NaN values (safe default for exponentiation).
   - **Outlier handling**: Replace infinite values with NaN, then with 0. Medians are more 
     robust to outliers than means, so extreme values in raw_proxy do not distort the scale factor.
   - **Preference hierarchy**: `ln_mean_NTL` is preferred because it represents the mean 
     of NTL observations; `ln_sum_NTL` (sum) is second choice; `mean_PM25` is a fallback.
   - **Scaling by medians**: Median is a robust statistic resistant to outliers. The ratio 
     of medians preserves relative ordering within the balanced dataset while aligning 
     magnitudes with the VIIRS distribution.
   - **Result**: New rows are on the same numeric scale as existing rows, helping the 
     downstream model treat all data consistently.

4. Filtering invalid rows
   - **Criteria**: Rows where `.geo` (GeoJSON) is empty or invalid are removed.
   - **Method**: Drop rows with empty `.geo` field before concatenation.
   - **Rationale**: The model requires valid point geometries; rows without valid 
     coordinates cannot be used for training or prediction.

5. Output format and concatenation
   - The script creates a DataFrame with columns: `system:index`, `avg_rad`, `.geo`.
   - `.geo` is a JSON string in GeoJSON Point format: `{"geodesic": false, "type": "Point", "coordinates": [lon, lat]}`.
   - The balanced prepared data is concatenated with the existing `data.csv` using `pd.concat()`.
   - The merged result is written to `data_prepared.csv`.
   - The project `model.py` constant `DATA_PATH` has been updated to point to 
     `data_prepared.csv` so the training pipeline uses the prepared dataset by default.

Data Quality Techniques Applied
--------------------------------

1. **Null/Missing Value Handling**:
   - NaN values in `ln_mean_NTL`, `ln_sum_NTL`, and `mean_PM25` are replaced with 0 before exponentiation.
   - Rows where centroid computation fails (no valid geometry) are assigned (NaN, NaN) coordinates.
   - Rows with invalid geometries are filtered out before final concatenation.

2. **Outlier Handling via Robust Statistics**:
   - Medians (instead of means) are used to compute the scaling factor, making the 
     transformation resistant to extreme values in either dataset.
   - Infinite values (from log inversions) are explicitly replaced with NaN and then 0.

3. **No Duplicate Removal**:
   - Duplicates are intentionally preserved because each polygon (or VIIRS point) is 
     a unique spatial observation. Removing duplicates could bias the training set.

4. **No Manual Outlier Removal**:
   - The median-based scaling approach naturally mitigates the influence of outliers 
     without explicit removal. If fine-grained outlier detection is desired, it should 
     be done in a later feature engineering step, not during merge.

5. **Column Validation**:
   - Existing data (`data.csv`) is validated to contain required columns (`avg_rad`, `.geo`).
   - Missing columns cause an informative error instead of silent failure.

Design rationale and trade-offs
--------------------------------

- **Using NTL-derived fields as a proxy for radiance**: Night-time lights are 
  conceptually related to artificial light emissions and therefore provide a 
  reasonable proxy when direct VIIRS radiance is unavailable. Inverting the 
  logged NTL and scaling by medians is a pragmatic way to align magnitudes while 
  preserving relative structure. The exponential (log-inversion) recovers the 
  original magnitude on a natural scale, which is necessary for the gradient 
  boosting model to interpret the feature correctly.

- **Centroid approximation by averaging vertices**: This avoids adding heavy GIS 
  dependencies (e.g., `shapely` or `geopandas`) and is acceptable for 
  coarse-grained analysis. The method is deterministic and reproducible. 
  If precise centroids or area-weighted centroids are required, a geometry-aware 
  library should be used.

- **Scaling by medians (robust to outliers)**: Using medians reduces the influence 
  of extreme NTL values that could distort the mapping. Unlike the mean, the 
  median is resistant to a small number of very large or very small outliers. 
  This is appropriate when the datasets may have measurement errors or extreme 
  environmental conditions.

- **No explicit duplicate removal**: Each polygon and VIIRS point is a unique 
  spatial observation. Removing duplicates could bias the training set toward 
  less-represented geographic areas. If true duplicates (identical coordinates and 
  values) exist, they are preserved to maintain data fidelity.

- **No manual outlier removal during merge**: The approach intentionally avoids 
  aggressive outlier removal at the merge stage because:
  - Extreme values may be valid (e.g., cities with very high NTL).
  - The median-based scaling naturally mitigates outlier influence.
  - Outlier detection and removal should be a deliberate step in feature engineering, 
    not an implicit consequence of data merging.
  - Models like Gradient Boosting are relatively robust to outliers in numeric features.

How to reproduce
----------------
1. Create and activate a Python environment that satisfies `requirements.txt`.
2. Run the preprocessing script from the project root:

```bash
python -m scripts.preprocess_balanced
```

3. This writes `data_prepared.csv` and `model.py` is already set to use it. Start the Flask app as usual (`python app.py`) to train and evaluate using the prepared data.

Next steps / Recommendations
----------------------------
- Validate the proxy mapping: inspect the distributions of `avg_rad` before and after merging (`data.csv` vs `data_prepared.csv`) to ensure the scaled NTL proxy behaves as intended.
- Consider re-running Phase 2 feature engineering to include new pollutant and socio-economic features from the balanced dataset (e.g., `mean_PM25`, `ln_GDPpc`) rather than reducing them to a single `avg_rad` proxy. That would likely improve model performance and interpretability.
- If higher spatial fidelity is required, re-compute centroids with `shapely.geometry.shape` or use `geopandas` to preserve projection and exact polygon centroids.

Contact
-------
If you want a different mapping for `avg_rad` (for example, a learned calibration using overlapping points), tell me and I can implement a regression-based mapper using overlapping samples between datasets.
