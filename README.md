# Machine Learning Energy Disaggregation – Fridge Prediction

This project predicts fridge power consumption from aggregated household smart-meter data (Chain2 protocol).  
The structure allows multiple teammates to work on independent models while sharing a unified preprocessing pipeline.

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
