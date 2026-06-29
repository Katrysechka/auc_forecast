# AUC Forecast — прогнозирование Reach & Frequency для VK Ads

Сравнение четырёх методов прогнозирования R&F-кривых на датасете VK Ads Auction
Forecasting. Для каждой кампании предсказываются три монотонно убывающих доли аудитории:
видевшая объявление **≥ 1**, **≥ 2**, **≥ 3** раза.

## Метрика

**SMLAR** — Smoothed Mean Log Accuracy Ratio, ε = 0.005:

```
SMLAR(y, ŷ) = mean(|log((ŷ + ε) / (y + ε))|) × 100 %
```

Симметрична по знаку ошибки, работает в логарифмической шкале (адекватной для долей
0–1), устойчива к малым `y` за счёт ε. Чем меньше — тем лучше.

Параллельно отслеживается **monotonicity violation** — доля кампаний, на которых
предсказание нарушает `y1 ≥ y2 ≥ y3`. Это формальная ошибка, даже если SMLAR хороший.

## Единый leak-safe протокол

Все методы используют одно и то же:

- **Time-based holdout 80/20** — сортировка по `(hour_start, row_index)`, последние 202
  кампании в holdout, первые 806 в train. Защищает от temporal leakage.
- **5-fold campaign CV на train** (seed = 42, `np.random.default_rng`).
- **Per-campaign cutoff фичи** — все аудиторные и паблишерские агрегаты строятся только
  по строкам истории `hour < hour_start_j`. 39 фичей.
- **Split Conformal Prediction** α = 0.10, per-target независимо.

## Результаты

| Метод | CV SMLAR % | Holdout SMLAR % | Mono viol. CV % | CP coverage |
|---|---|---|---|---|
| catboost-default | 47.59 ± 5.49 | 40.88 | 4.72 | 0.896 |
| catboost-optuna | 41.24 ± 4.95 | 33.66 | 6.95 | 0.890 |
| **catboost-log_link (MultiRMSE on log(y+ε))** | **26.45 ± 0.97** | **22.88** | 4.34 | — |
| **mlp-smlar_smooth** | **29.17 ± 1.99** | **22.63** | **0.00** | 0.908 |
| **monte_carlo + β** | **28.24 ± 1.56** | **23.85** | **0.00** | 0.906 |
| **set_transformer-A_full** | **27.53 ± 2.12** | **21.21** | **0.00** | — |

### Ключевые наблюдения

- **Set Transformer — лучший по Holdout** (21.21 %) при **0 % нарушений монотонности
  по построению** (softmax-голова + tail-cumsum математически гарантируют).
- **Proxy-gap MSE → SMLAR-compatible loss работает на ОБЕИХ архитектурах.**
  На MLP: MSE → SMLAR-smooth даёт −38.7 п.п. на CV (67.86 → 29.17). На CatBoost:
  default MultiRMSE → MultiRMSE на `log(y+ε)` даёт −21.1 п.п. на CV (47.59 → 26.45).
  Смена функции потерь / шкалы таргета — самый большой эмпирический рычаг проекта.
- **CatBoost с log-link догоняет лидеров** (26.45 % CV / 22.88 % Holdout) — на уровне
  Set Transformer и лучше mlp-smlar_smooth по CV. Старое утверждение «табличные
  деревья в принципе не дотягивают» — артефакт неподходящей шкалы таргета, а не
  ограничение деревьев. **Кастомный SMLAR-loss через `RMSEWithUncertainty` / 3
  независимых регрессора у CatBoost проваливается** (173.87 % CV) — CatBoost 1.2.x не
  поддерживает кастомные multi-target objectives; рабочая инкарнация SMLAR-loss для
  деревьев — именно target-transform (log-link), а не custom objective.
- **Naive Monte Carlo даёт SMLAR ≈ 396 %** — параметрическое NegBinomial-предположение
  на поюзерных частотах неадекватно. β-CatBoost-калибратор вытаскивает оценку до 28 %.
  Содержательный вывод: интерпретируемость NegBinomial — фикция, в проде это
  ML-калибровка над дешёвой генеративной заготовкой.
- **Четыре метода кучно** (Set Transformer 27.5 / catboost-log_link 26.5 /
  mlp-smlar_smooth 29.2 / MC+β 28.2 по CV) — разница в пределах std. Преимущество
  Set Transformer — конструктивная гарантия монотонности (0 %, не «повезло») плюс
  побочный продукт в виде явного распределения по reach-curve `P(X = k)`. Преимущество
  catboost-log_link — самая дешёвая обучаемость (~6 с против 5 × 18.8 мин на GPU).

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

jupyter notebook notebooks/method_1_catboost/   
jupyter notebook notebooks/method_2_mlp/        
jupyter notebook notebooks/method_3_monte_carlo/
python notebooks/method_4_set_transformer/build_results.py

pytest tests/

streamlit run streamlit_app/app.py
```

## Структура репозитория

```
auc_forecast/
├── core/                
│   ├── config.py            EPS=0.005, SEED=42, MC_N_SIMS=500, ALPHA=0.10, пути
│   ├── data_io.py           load_raw, load_validate_full
│   ├── splits.py            time-based 80/20 + 5-fold campaign CV
│   ├── leak_safe_features.py UserHistoryIndex + PublisherHistoryIndex + N-feat builder
│   ├── metrics.py           SMLAR + monotonicity_violation + 6 torch-лоссов
│   ├── conformal.py         SplitCP / MultiTargetSplitCP
│   ├── ci.py                ConformalRunner-обёртка
│   ├── runner.py            load_unified_data, Method-протокол
│   └── reporting.py       
│
├── notebooks/            
│   ├── method_1_catboost/      01_data_and_features / 02_train_default_and_optuna / 03_evaluation_and_results
│   ├── method_2_mlp/           01_setup_and_architecture / 02_train_six_losses / 03_evaluation_and_cp
│   ├── method_3_monte_carlo/   01_naive_mc_and_convergence / 02_beta_correction_training / 03_holdout_cp_credible
│   └── method_4_set_transformer/
│       ├── 01_setup_and_architecture / 02_train_ablation / 03_evaluation_and_results
│       ├── src/                 model, data, train, leak_tests
│       └── build_results.py    
│
├── results/
│   ├── per_method/*.json   
│   ├── per_target_summary.json  
│   └── figures/             
│
├── streamlit_app/app.py   
├── model/                 
├── data/                  
├── parquet_files/         
├── tests/                 
└── README.md              
```
