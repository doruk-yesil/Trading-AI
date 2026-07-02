# Veri Komutları

```
python scripts/download_yahoo_finance.py     # günlük veri (data/raw/)
python scripts/download_hourly_data.py       # saatlik veri (data/hourly/)
python scripts/build_features.py             # feature + hedef üret (data/processed/)
python scripts/daily_refresh.py              # yukarıdaki üçünü tek seferde yapar
```

# Model Komutları

```
python scripts/train_model.py --models linear random_forest   # baseline grubu (taban)
python scripts/train_model.py --models xgboost lightgbm        # GBM grubu (asıl modeller)
# neural modeller (LSTM/Transformer) — henüz implement edilmedi, Phase 4 (ileride)
```

```
python scripts/predict.py                    # kayıtlı en son modelle tahmin üret (train etmez)
```
