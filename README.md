# Machine Learning Energy Disaggregation – Fridge Prediction

This project predicts fridge power consumption from aggregated household smart-meter data (Chain2 protocol).  
The structure allows multiple teammates to work on independent models while sharing a unified preprocessing pipeline.

HOW TO RUN IN TERMINAL:
-PREPROCESSING:
  python -m src.common.resampling
  python -m src.datasets.make_train
  python -m src.datasets.make_test
  IF YOU WANT SMOOTHED DATA:
    python -m src.datasets.make_light_smooth
    or python -m src.datasets.make_train_smooth
-FINAL MODEL SUBMITTED:
  Run the jupyter notebook under src/FINAL MODEL/
-OTHER EXPERIMENTED MODELS
  python -m src.experiments.{model}.test
  python -m src.experiments.{model}.predict

For the experiments the submissions are returned under submission/.
For the FINAL MODEL the submission is ASLO UNDER SUBMISSION/.

ABOUT THE PROJECT:
In this project we tried to estimate fridge power usage from household aggregate power readings. Most of the work ended up being about getting the data into a form that models could learn from.

Preprocessing:
We built a preprocessing pipeline that:
-parsed and aligned timestamps
-resampled all signals to a 1-minute grid
-handled missing values
-cleaned unrealistic spikes
-and engineered features like rolling averages, lags, and time-based indicators.
We also smoothed the fridge target so the model wasn’t learning raw noise.

Classical ML Models:
We trained several models, mainly LightGBM and XGBoost, on the engineered features.
-These gave the most reliable results and consistently landed around the 24–25 MAE range on the leaderboard.

Other Experiments:
-We tried more ambitious ideas:
-ON/OFF classification before regression
-hybrid models
-LOHO (leave-one-home-out) approaches
-deep learning models like TCNs
-and multiple windowing strategies for sequence learning.

Most of these experiments looked good on internal validation but ended up overfitting and didn’t improve the leaderboard performance.

Smoothing & Ensembling

We also tested smoothing predictions and averaging outputs from multiple models.
These helped stabilize predictions but didn’t significantly break past the ~24 MAE barrier.

---

## 📁 Project Structure

```text
hackaton/
  data/
    raw/
      train.csv
      test.csv
    processed/
      train_1min.csv
      test_1min.csv
      train_features.parquet
      test_features.parquet

  src/
    common/
      config.py
      resampling.py
      features.py
      evaluation.py

    datasets/
      make_train.py
      make_test.py

    experiments/
      tudor_lgbm/
        train.py
        predict.py
        config.py
      alice_cnn/
      bob_baseline/

    utils/
      logging.py

  models/
  notebooks/
  submission/
  requirements.txt
  README.md
