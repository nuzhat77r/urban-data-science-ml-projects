# Urban Mobility Forecasting & Building Footprint Segmentation

Two applied machine-learning projects exploring **predictive modeling of urban mobility** and **semantic segmentation of aerial imagery**. The work covers exploratory analysis, statistical modeling, feature engineering, deep learning, leakage-aware evaluation, and interpretation of model errors.

## Projects at a Glance

| Project | Goal | Main approach | Key result |
|---|---|---|---|
| Capital Bikeshare Demand Forecasting | Predict hourly bicycle rentals | Negative Binomial regression and LightGBM | **R² = 0.903**, **RMSLE = 0.415** on held-out hours (primary no-target-lag model) |
| Building Footprint Segmentation | Identify building pixels in aerial images | U-Net with pretrained ResNet34 encoder | **IoU = 0.686**, **Dice = 0.814** on a fully held-out city |

## 1. Capital Bikeshare Demand Analysis & Forecasting

**Notebook:** [`Part1_Capital_Bikeshare_v1_0.ipynb`](Part1_Capital_Bikeshare_v1_0.ipynb)

### Objective
Understand how time, weather, and rider type affect bicycle rental demand in Washington, D.C., and forecast total hourly rentals.

### Data
- [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **17,379 hourly records** covering 2011–2012
- Calendar, weather, and rental-count variables, including casual and registered riders

### Workflow
1. **Exploratory data analysis:** Examine hourly, weekday/weekend, seasonal, and weather-related demand patterns; identify missing timestamps and unusual demand periods.
2. **Statistical analysis:** Diagnose multicollinearity between temperature and apparent temperature; compare ordinary least squares, Poisson, and Negative Binomial models for rental counts.
3. **Feature engineering:** Use calendar and weather predictors with cyclical encodings; exclude casual and registered counts from total-demand forecasting because they directly reveal the target.
4. **Forecasting:** Train a Negative Binomial baseline and LightGBM models.
5. **Evaluation:** Train on days **1–20** of each month and evaluate on days **21–end** of each month, across 24 monthly windows. The primary model does not use target-derived lag features from the evaluation period.

### Findings
- Working days show commuting peaks near **8 AM** and **5–6 PM**; weekends and holidays show broader midday activity.
- Casual riders are more sensitive to working-day and temperature differences than registered riders.
- Rental counts are overdispersed, supporting Negative Binomial regression over a simple Poisson specification.
- Temperature and apparent temperature are highly correlated, so apparent temperature was removed from the primary regression specification.

### Forecasting results

| Model | MAE | RMSE | RMSLE | R² |
|---|---:|---:|---:|---:|
| Seasonal naive (rolling reference) | 58.160 | 101.672 | 0.651 | 0.685 |
| Negative Binomial GLM (weather + calendar) | 66.261 | 106.132 | 0.664 | 0.657 |
| **LightGBM (primary; no target lags)** | **33.256** | **56.274** | **0.415** | **0.903** |
| LightGBM (supplementary rolling-lag reference) | 22.351 | 36.559 | 0.292 | 0.959 |

> **Evaluation note:** The rolling-lag reference assumes updated rental observations are available during the evaluation window. It is not directly equivalent to the primary forecast, which uses calendar and weather information without observed target lags. Weather covariates are assumed to be available at forecast time.

## 2. Building Footprint Segmentation from Aerial Imagery

**Notebook:** [`Part2_Inria_Building_Segmentation_v1_0.ipynb`](Part2_Inria_Building_Segmentation_v1_0.ipynb)

### Objective
Extract building footprints from high-resolution aerial images using binary semantic segmentation, and evaluate generalization to an unseen geographic area.

### Data
- [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
- **180 labeled RGB tiles**, each **5000 × 5000 pixels**
- Cities: Austin, Chicago, Kitsap, Tyrol, and Vienna
- Binary masks distinguishing buildings from background

### Workflow
1. **Leakage-aware partitioning:** Split at the full-tile level before extracting patches; hold out all **36 Tyrol tiles** for final testing. Use **116 training** and **28 validation** tiles from the other four cities.
2. **Preprocessing and augmentation:** Use **512 × 512** RGB patches, ImageNet normalization, and spatial augmentations.
3. **Architecture:** Train a **U-Net** with an **ImageNet-pretrained ResNet34 encoder**.
4. **Loss:** Combine binary cross-entropy and Dice loss to address pixel-level classification and foreground imbalance.
5. **Checkpoint selection:** Select the checkpoint with the highest validation IoU (**epoch 13**, validation IoU **0.760**, Dice **0.864**).
6. **Full-image inference:** Stitch sliding-window predictions with **512-pixel windows** and **384-pixel stride**, averaging probabilities in overlapping areas.
7. **Evaluation:** Measure IoU, Dice/F1, precision, recall, and pixel accuracy; inspect masks and error maps.

### Results on unseen Tyrol tiles

| Metric | Global pixel-pooled score | Per-tile mean ± SD |
|---|---:|---:|
| IoU | **0.686** | 0.687 ± 0.064 |
| Dice / F1 | **0.814** | 0.813 ± 0.045 |
| Precision | 0.760 | 0.764 ± 0.086 |
| Recall | 0.875 | 0.876 ± 0.028 |
| Pixel accuracy | 0.977 | 0.977 ± 0.014 |

### Error analysis
The model identifies most building pixels, but its **recall is higher than its precision**. Some roof regions are over-expanded or merged, and small or isolated buildings can be difficult to segment. Possible next steps include boundary-aware losses, stronger augmentation, and post-processing to improve footprint precision.

## Repository Structure

```text
.
├── README.md
├── Part1_Capital_Bikeshare_v1_0(1).ipynb
└── Part2_Inria_Building_Segmentation_v1_0(1).ipynb
```

## Running the Notebooks

1. Download the respective datasets from the official links above.
2. Open the notebooks in **Google Colab** or a compatible Jupyter environment.
3. Install the dependencies referenced in each notebook and adjust local/Google Drive dataset paths as needed.
4. Run the cells in order. The segmentation workflow benefits from a GPU and requires substantial storage for full-resolution imagery.

**Main tools:** Python, pandas, NumPy, Matplotlib, seaborn, statsmodels, scikit-learn, LightGBM, PyTorch, and image segmentation libraries.

## Notes on Reproducibility

- Large datasets and model checkpoints are not included in this repository by default; download data from the original sources.
- Forecasting results are based on the specified within-month train/evaluation windows, not an entirely future-year holdout.
- Segmentation test results are based on a geographically held-out city, rather than randomly mixed image patches.

## Project Summary

Together, these projects demonstrate practical approaches to **statistical inference, forecasting, computer vision, experimental design, and model evaluation**, with particular attention to preventing data leakage and interpreting performance beyond a single metric.
