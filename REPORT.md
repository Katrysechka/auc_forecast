# Внутренний отчёт — структура, состояние, ограничения

## 1. О чём этот репозиторий

**Задача.** Прогнозирование Reach & Frequency на датасете VK Ads Auction Forecasting:
1008 рекламных кампаний, 27 769 пользователей, 1,15 млн исторических показов. Для каждой
кампании предсказываются три таргета:

- `at_least_one` — доля аудитории, увидевшая объявление **≥ 1 раз**
- `at_least_two` — доля, увидевшая **≥ 2 раз**
- `at_least_three` — доля, увидевшая **≥ 3 раз**

Эти три числа монотонно убывают (`y1 ≥ y2 ≥ y3` по построению). Если модель нарушает
это неравенство — это формальная ошибка, даже если SMLAR хороший. Поэтому monotonicity
violation отслеживается как отдельная метрика.

**Метрика.** SMLAR (Smoothed Mean Log Accuracy Ratio) с ε = 0.005:

```
SMLAR(y, ŷ) = mean(|log((ŷ + ε) / (y + ε))|) × 100%
```

Чем меньше — тем лучше. Симметричная по знаку ошибки, устойчивая к малым `y` за счёт ε,
работает в логарифмической шкале (адекватной для долей в диапазоне 0–1).

**Сравниваются 4 метода** на едином leak-safe протоколе:

1. **CatBoost** — четыре варианта: `default` MultiRMSE (бейзлайн), `optuna` (25 trials,
   MultiRMSE), `log_link` (MultiRMSE на `log(y+ε)` — target-transform-инкарнация
   SMLAR-совместимого обучения) и `smlar_smooth` (кастомный SMLAR-loss через
   3 независимых регрессора — проваливается, CatBoost 1.2.x не поддерживает custom
   multi-target objectives). `log_link` — фактически лучший по CV среди всего проекта.
2. **MLP × 6 функций потерь** (`mse / mae / huber / msle / rmsle / smlar_smooth`) — при
   фиксированной архитектуре изолирует эффект SMLAR-совместимой функции потерь
   (количественно показывает proxy-gap между MSE-обучением и целевой метрикой).
3. **Monte Carlo + β-коррекция через CatBoost** — поюзерное сэмплирование частот из
   NegBinomial с `S = 500` симуляций; затем поправочный CatBoost обучается на лог-отношении
   `log(y_true / y_naive_mc)` и калибрует наивную оценку.
4. **Set Transformer** (ISAB + PMA, Lee et al. 2019) с монотонной distributional-головой
   и SMLAR-smooth лоссом — методическая новизна работы. Цифры заморожены из корректного
   прогона на Colab GPU (см. §5 про CSV ↔ log).

Доверительные интервалы для **всех четырёх** методов — единая обёртка Split Conformal
Prediction (α = 0.10, per-target независимо).

---

## 2. Структура папок

```
auc_forecast/
├── data/                          # сырые TSV (users / history / validate / validate_answers)
├── parquet_files/                 # хранится только не-leaky validate-агрегат
├── model/                         # текущие лучшие веса: best_catboost_optuna.cbm, best_mlp_smlar_smooth.pt
│
├── core/                          # ОБЩАЯ инфраструктура (единственный источник правды)
│   ├── config.py                  #   EPS=0.005, SEED=42, MC_N_SIMS=500, ALPHA=0.10, пути
│   ├── data_io.py                 #   load_raw, load_validate_full
│   ├── splits.py                  #   time-based 80/20 + 5-fold по индексам кампаний
│   ├── leak_safe_features.py      #   UserHistoryIndex + PublisherHistoryIndex (префикс-суммы)
│   │                              #   + build_leak_safe_aggregates (39 колонок, per-campaign cutoff)
│   ├── metrics.py                 #   SMLAR + monotonicity_violation + 6 torch-лоссов
│   ├── conformal.py               #   SplitCP / MultiTargetSplitCP
│   ├── ci.py                      #   ConformalRunner (оборачивает любой fit/predict-регрессор)
│   ├── runner.py                  #   load_unified_data, run_all_methods, Method-протокол
│   └── reporting.py               #   схема results.json, рендер per_method/*.md, final_comparison
│
├── notebooks/                     # ОДНА ПАПКА НА МЕТОД (ноутбуки + код метода)
│   ├── method_1_catboost/
│   │   ├── 01_data_and_features.ipynb
│   │   ├── 02_train_default_and_optuna.ipynb
│   │   └── 03_evaluation_and_results.ipynb
│   ├── method_2_mlp/
│   │   ├── 01_setup_and_architecture.ipynb     # ImprovedMLP (256→128→64+Sigmoid)
│   │   ├── 02_train_six_losses.ipynb           # все 6 лоссов; победитель → model/best_mlp_smlar_smooth.pt
│   │   └── 03_evaluation_and_cp.ipynb
│   ├── method_3_monte_carlo/
│   │   ├── 01_naive_mc_and_convergence.ipynb   # векторизованный NegBinomial sampler
│   │   ├── 02_beta_correction_training.ipynb   # β-CatBoost на log(y_true / y_naive_mc)
│   │   └── 03_holdout_cp_credible.ipynb        # CP + parametric credible coverage
│   └── method_4_set_transformer/               # бывшая direction_2_set_transformer/
│       ├── 01_setup_and_architecture.ipynb
│       ├── 02_train_ablation.ipynb             # запускается на Colab GPU
│       ├── 03_evaluation_and_results.ipynb     # локальный re-run holdout из frozen-весов
│       ├── src/                                # model, data, train, leak_tests
│       ├── results_frozen/                     # ЗАМОРОЖЕННЫЕ ablation summary + folds (источник правды)
│       ├── results/                            # _latest при локальных перезапусках (gitignored)
│       └── build_results.py                    # пересобирает results.{json,md} из results_frozen/
│
├── results/                       # АГРЕГИРОВАННЫЕ результаты, схема едина для всех методов
│   ├── per_method/                #   по одному JSON+MD на метод-победитель семейства
│   │   ├── catboost.{json,md}
│   │   ├── mlp.{json,md}              # winner + все 6 вариантов в extra.variants
│   │   ├── monte_carlo.{json,md}
│   │   └── set_transformer.{json,md}  # winner A_full + 5 ablation в extra.variants
│   ├── per_target_summary.json    #   per-target breakdown по победителю каждого семейства
│   ├── final_comparison.md        #   сводная таблица всех вариантов
│   ├── final_comparison.csv
│   └── figures/                   #   сгенерированные графики
│
├── tests/                         # 33 теста, все проходят
│   ├── test_metrics.py            # SMLAR, eps, монотонность, coverage
│   ├── test_conformal.py          # SplitCP / MultiTargetSplitCP
│   ├── test_splits.py             # детерминизм 806/202, tiebreaker, разбиение фолдов
│   ├── test_leak_safe_features.py # инвариантность per-cutoff, cold-start, 39 колонок
│   ├── test_ci_coverage.py        # синтетический Гаусс → эмпирическое покрытие ≈ 0.90
│   ├── test_runner_protocol.py    # два вызывающих кода получают одинаковые split-ы
│   └── test_no_method_imports_in_core.py  # AST-страж, нет циклических импортов
│
├── streamlit_app/app.py           # демо-дашборд: 3 вкладки (О проекте / Методы / Итоговое сравнение)
│                                  # Plotly, без matplotlib; все цифры читаются из results/
├── thesis/                        # LaTeX-исходники работы — отдельная зона
│
├── README.md                      # короткая точка входа: задача, цифры, быстрый старт
└── REPORT.md                      # ЭТОТ ФАЙЛ — полный внутренний отчёт
```

---

## 3. Единый протокол валидации

Все методы получают **один и тот же объект** через `core.runner.load_unified_data()` и
используют **одно и то же разбиение**:

### 3.1 Time-based holdout 80/20

Кампании сортируются по `(hour_start, row_index)` — последние 20% (202 кампании) уходят
в holdout, остальные 80% (806) — в train. Tiebreaker по `row_index` нужен, потому что в
датасете много кампаний с одинаковым `hour_start`, и без явного второго ключа сортировки
порядок неопределён → недетерминированный split.

**Почему time-based, а не random:** R&F-задача = предсказание на будущих кампаниях.
Random-split позволяет фичам кампаний из «будущего» утечь в train через usage history
пользователя, что артифициально занижает CV-ошибку. Time-based это исключает.

### 3.2 5-fold campaign CV на train

806 индексов кампаний перемешиваются через `np.random.default_rng(42).shuffle`, затем
делятся на 5 непрерывных фолдов. Каждая кампания попадает в validation ровно одного фолда.

### 3.3 Per-campaign cutoff фичей

Для каждой кампании `j` все аудиторные и паблишерские агрегаты считаются только по
строкам истории, где `hour < hour_start_j`. Векторизованная реализация — в
`core/leak_safe_features.py`: общие `UserHistoryIndex` и `PublisherHistoryIndex` с
префикс-суммами позволяют перестроить фичи на кампанию за `O(V·log N)` вместо `O(N·V)`.

Без этой оптимизации регенерация фичей для 5 фолдов × 4 методов заняла бы часы.

### 3.4 Inner train/calibration split для Split CP

Внутри каждого outer-фолда: seeded shuffle 75/25 → 75% используется для дообучения
модели после CV-выбора параметров, 25% — для калибровки конформальных квантилей.
Per-target SplitCP при α = 0.10 выдаёт **три независимых** интервала для `y1, y2, y3`.

### 3.5 Унифицированная схема results.json

Каждый метод пишет одну и ту же структуру:

```json
{
  "method": "catboost-optuna",
  "split_protocol": "time-based 80/20 + 5-fold campaign CV (seed=42)",
  "leak_safe": true,
  "features": "leak-safe 39-feat aggregates",
  "cv": {
    "smlar_flat_mean": 43.55,
    "smlar_flat_std": 5.88,
    "smlar_per_target": {"y1": 41.2, "y2": 44.8, "y3": 44.6},
    "monotone_viol_mean": 0.0645
  },
  "holdout": {"smlar_flat": 34.49, "monotone_viol": 0.062},
  "ci": {
    "type": "split_cp",
    "alpha": 0.10,
    "coverage_per_target": {"y1": 0.91, "y2": 0.90, "y3": 0.89},
    "width_mean": 0.085
  },
  "source": "fresh run @ <git_sha>"
}
```

`core/runner.py` агрегирует все эти JSON-ы в единую таблицу
`results/final_comparison.{md,csv}`.

---

## 4. Что такое CV SMLAR и Holdout SMLAR

Два числа стоят в каждой строке итоговой таблицы; ниже — что они означают и почему
смотреть нужно на оба.

### 4.1 CV SMLAR

**Определение.** Среднее SMLAR по 5 фолдам кросс-валидации на train (806 кампаний).
В каждом фолде модель обучается на 4/5 train, метрика считается на оставшейся 1/5;
после 5 фолдов берётся среднее и стандартное отклонение.

**Что показывает.**
- Точечную оценку обобщающей способности модели.
- Неопределённость оценки через std по фолдам — это и есть видимая в таблице ± величина.
- Является основой выбора гиперпараметров (Optuna для CatBoost ориентируется на CV).

**Свойства на нашем датасете.**
- Каждая кампания побывала в validation ровно один раз.
- Train и validation в фолде могут содержать кампании из перекрывающихся временных
  отрезков, поэтому CV-оценка чуть **оптимистичнее**, чем будущая производительность.

### 4.2 Holdout SMLAR

**Определение.** SMLAR на 202 отложенных кампаниях (последние 20% по `hour_start`).
Модель обучается на **всём** train (806) с лучшими гиперпараметрами, найденными по CV,
и применяется к holdout — **один раз**.

**Что показывает.**
- Честную оценку «в будущем»: holdout-кампании по времени идут **после** train.
- Финальное число для сравнения методов (на нём не подбирается ничего).

**Почему оно полезно отдельно от CV.**
- CV использовался для подбора гиперпараметров — туда мог утечь сигнал через
  оптимизацию (особенно для Optuna с 25 trials).
- Holdout этого избегает: модель его не видела и решения по нему не принимались.

### 4.3 Разница между ними в наших результатах

| Метод | CV SMLAR | Holdout SMLAR | Разница |
|---|---|---|---|
| catboost-default | 47.59% | 40.88% | −6.71 |
| catboost-optuna | 41.24% | 33.66% | −7.58 |
| catboost-log_link | 26.45% | 22.88% | −3.57 |
| mlp-smlar_smooth | 29.17% | 22.63% | −6.54 |
| monte_carlo+β | 28.24% | 23.85% | −4.39 |
| set_transformer-A_full | 27.53% | **21.21%** | −6.32 |

У всех методов Holdout **ниже** CV (т. е. лучше), потому что:

- Поздние кампании в среднем имеют больше истории для пользователей и паблишеров
  (`UserHistoryIndex.cpm_prefix` обрезается на более позднем `hour_start_j` → больше
  накопленных событий) → leak-safe фичи качественнее.
- Распределение поздних кампаний у́же по бюджету и длительности (хвост распределения
  «больших» кампаний остался в train) → задача чуть проще.

**Расхождение CV − Holdout само по себе диагностично.** У CatBoost-Optuna оно −7.6 п.п.
(больше, чем у других) — это сигнал слабого переобучения под CV через Optuna. У MC оно
наименьшее (−4.4 п.п.), что согласуется с тем, что β-коррекция малопараметрична и не
склонна переобучаться. CatBoost-log_link тоже даёт небольшой разрыв (−3.6 п.п.) — без
Optuna-тюнинга деревья меньше переобучаются под CV. Set Transformer (−6.3 п.п.) и
MLP-SMLAR-smooth (−6.5 п.п.) находятся примерно посередине.

---

## 5. Состояние по методам

| Метод | Источник | CV SMLAR | Holdout SMLAR | Mono viol (CV) | CP cov | Время прогона |
|---|---|---|---|---|---|---|
| **catboost-log_link** | локальный | **26.45% ± 0.97** | **22.88%** | 4.34% | — | ~6 с |
| set_transformer-A_full | frozen log + local holdout | **27.53% ± 2.12** | **21.21%** | **0.00%** | — | 5 × 18.8 мин на Colab GPU + ~3 мин на CPU |
| monte_carlo+β | локальный | 28.24% ± 1.56 | 23.85% | 0.00% (post-hoc sort) | 0.91 | ~6 мин |
| mlp-smlar_smooth (winner) | локальный | 29.17% ± 1.99 | **22.63%** | 0.00% | 0.91 | ~15 с на лосс |
| mlp-msle | локальный | 30.78% ± 1.44 | 24.27% | 1.49% | 0.90 | — |
| mlp-mae | локальный | 37.80% ± 1.67 | 26.78% | 2.48% | 0.92 | — |
| catboost-optuna | локальный | 41.24% ± 4.95 | 33.66% | 6.95% | 0.89 | ~5 мин (25 Optuna trials) |
| mlp-rmsle | локальный | 42.91% ± 2.69 | 32.54% | 2.23% | 0.93 | — |
| catboost-default | локальный | 47.59% ± 5.49 | 40.88% | 4.72% | 0.90 | ~8 с |
| mlp-huber | локальный | 50.55% ± 2.51 | 39.25% | 6.08% | 0.92 | — |
| mlp-mse | локальный | 67.86% ± 4.00 | 54.14% | 10.43% | 0.92 | — |
| catboost-smlar_smooth (custom loss, fail) | локальный | 173.87% ± 26.60 | — | — | — | ~16 с |

**Ablation Set Transformer** (5 вариантов, CV из заморожённого Colab-лога):

| Вариант | CV SMLAR | Attention | Distr.-head | Loss | Что вырубается |
|---|---|---|---|---|---|
| A_full | **27.53 ± 2.12** | ✓ ISAB | ✓ softmax + cumsum | smlar_smooth | (полная модель) |
| B_no_attn | 28.94 ± 2.30 | ✗ mean-pool | ✓ | smlar_smooth | attention |
| C_no_distr | 34.28 ± 10.88 | ✓ ISAB | ✗ MLP-head | smlar_smooth | distributional head |
| E_none | 61.26 ± 3.07 | ✗ mean-pool | ✗ MLP-head | mse | всё, базовый ablation |
| D_no_smlar | 81.86 ± 13.88 | ✓ ISAB | ✓ | mse | только loss |

### 5.1 Ключевые наблюдения

**MSE → SMLAR-совместимый loss/шкала: proxy-gap воспроизводится на ОБЕИХ архитектурах.**
На MLP: смена `mse` → `smlar_smooth` даёт −38.7 п.п. CV (67.86% → 29.17%) при
фиксированной архитектуре и фичах. На CatBoost: смена `MultiRMSE на y` →
`MultiRMSE на log(y+ε)` (target-transform, log-link) даёт −21.1 п.п. CV
(47.59% → 26.45%) на тех же гиперпараметрах дерева (`iterations=800`, `depth=6`,
`lr=0.05`). Это эмпирически подтверждает, что MSE/MultiRMSE оптимизируют не ту шкалу,
в которой считается SMLAR (линейную vs логарифмическую). Смена функции потерь / шкалы
таргета — самый большой эмпирический рычаг проекта, и он переносится между классами
моделей.

**CatBoost-log_link догоняет лидеров (26.45% CV / 22.88% holdout).**
На уровне Set Transformer (27.53/21.21) и mlp-smlar_smooth (29.17/22.63) — фактически
лучший по CV среди всех методов проекта. Старое утверждение «табличные деревья в
принципе не дотягивают» — артефакт неподходящей шкалы таргета, а не ограничение
деревьев. Минусы: monotone viol CV = 4.34% (не 0%) — log-link не даёт конструктивной
гарантии монотонности, нарушения нужно лечить post-hoc сортировкой.

**Кастомный SMLAR-loss для CatBoost проваливается (173.87% CV).** CatBoost 1.2.x не
поддерживает кастомные multi-target objectives; обёртка из трёх независимых
регрессоров с SMLAR-smooth-loss разваливается из-за невозможности корректно передать
градиент. **Рабочая инкарнация SMLAR-совместимого обучения для деревьев — именно
target-transform (log-link), а не custom objective.** Это методический результат,
который имеет смысл отдельно зафиксировать в работе.

**Set Transformer vs catboost-log_link vs MLP-SMLAR-smooth: 27.53 / 26.45 / 29.17 CV.**
Разница в пределах std (~2 п.п.). Однако Set Transformer даёт **0% нарушений
монотонности по построению** (а не повезло на сиде), и выдаёт явное распределение по
reach-curve `P(X = 0, 1, …, K-1, ≥ K)`, что даёт побочный продукт — интерпретируемую
частотную картину для рекламодателя. У catboost-log_link обратное преимущество —
тренировка ~6 секунд против 5 × 18.8 минут на Colab GPU.

**MC + β-коррекция конкурентна (28.24% CV, 23.85% holdout).**
Naive MC сэмплинг без коррекции даёт SMLAR ≈ 400% (т. е. полностью промахивается). Это
не «хорошая физика» — это утверждение, что параметрическое предположение NegBinomial на
поюзерных частотах неадекватно. β-CatBoost вытаскивает MC из этой ямы до 28%.
Содержательный вывод: интерпретируемость NegBinomial — фикция; в проде это
ML-калибровка над дешёвой генеративной заготовкой.

**Monotonicity violations: CatBoost-default/Optuna 4.7–7.0%, CatBoost-log_link 4.3%,
MLP-MSE 10.4%, MLP-SMLAR-smooth / MC+β / Set Transformer 0%.**
Это центральный методический аргумент в пользу Set Transformer как метода с
**конструктивной** гарантией: monotonicity не «обнаружена в эксперименте», а заложена в
архитектуру головы (softmax + tail-cumsum). У MLP-SMLAR-smooth 0% — это удача
оптимизации, не гарантия; у MC — post-hoc сортировка. Только Set Transformer
гарантирует монотонность математически.

---

## 6. Set Transformer: разрешение конфликта CSV ↔ log

В пре-рефакторном репозитории файл `direction_2/results/ablation_summary.csv` содержал
`A_full = 85.54%` со `std = 27.36`, что противоречит `logs/ablation.log = 27.53% ± 2.12`.

**Причина расхождения.** Лог записывает полный 5-fold прогон (~60 мин на Colab GPU,
все 5 фолдов на полных данных). CSV — результат более позднего debug-перезапуска
(`FAST_DEBUG = True`, 2 фолда, ~5 мин), который перезаписал файл. Лог — источник правды.

**Решение.**

1. Корректные цифры **заморожены** в:
   - `notebooks/method_4_set_transformer/results_frozen/ablation_summary_frozen.csv`
   - `notebooks/method_4_set_transformer/results_frozen/ablation_folds_frozen.csv`

   Эти файлы закоммичены и считаются неизменяемыми.

2. Локальные перезапуски ноутбука `02_train_ablation.ipynb` пишут в
   `notebooks/method_4_set_transformer/results/`, что лежит в `.gitignore`.
   `core/runner.py` и `build_results.py` читают **только** `_frozen` версию.

3. `build_results.py` пересобирает `results/per_method/set_transformer.{json,md}` из
   `_frozen` каждый раз.

4. **Holdout** для A_full дополнительно посчитан локально (CPU ~3 мин) через
   `03_evaluation_and_results.ipynb` — не из лога, а из заморожённых весов модели —
   и записан в `holdout.smlar_flat = 21.21%`.

**Как обновить frozen после нового Colab-прогона:**

```
# 1. На Colab GPU
запустить notebooks/method_4_set_transformer/02_train_ablation.ipynb
# 2. Локально, после копирования CSV из Colab
cp notebooks/method_4_set_transformer/results/ablation_summary.csv \
   notebooks/method_4_set_transformer/results_frozen/ablation_summary_frozen.csv
python notebooks/method_4_set_transformer/build_results.py
```

---

## 7. Обоснование `MC_N_SIMS = 500`

Параметр зафиксирован в `core/config.py` со следующими аргументами:

**Эмпирически.** На предварительном convergence-эксперименте (старый ноутбук 05) SMLAR
стабилизируется при `S ≥ 500`: дальнейшее удвоение до `S = 1000` сдвигает оценку менее
чем на 0.5 п.п. на каждом из трёх таргетов, что меньше fold-to-fold std (1.56 п.п.).

**Теоретически.** Стандартная ошибка Монте-Карло оценки доли `p` по `S` сэмплам:
`SE = sqrt(p(1-p)/S)`. При типичных рейтах `p ∈ [0.1, 0.5]` и `S = 500`:
`SE ≈ 0.0224`. Это ниже ε = 0.005? Нет, выше. **Но** SMLAR работает в логарифмической
шкале с ε-смягчением, и относительная ошибка `SE/p` при `p = 0.3, S = 500` составляет
≈ 7.5%, что меньше fold-to-fold std (~5–8%) → дальнейшее увеличение `S` не уменьшит
доминирующий источник дисперсии.

**Cost.** S = 500 даёт время прогона ≈ 6 мин на полном датасете. S = 1000 удвоил бы
время и не улучшил точечную оценку.

Параметр `MC_N_SIMS_HIGH = 1000` оставлен в config'е на случай построения **узкого**
credible-interval'а (там `S` сильнее влияет на ширину).

---

## 8. Известные ограничения

**Holdout для Set Transformer — только для A_full.** Holdout пересчитан локально из
заморожённых весов для победителя (A_full → 21.21%). Для остальных четырёх вариантов
ablation holdout не считался: ablation проводится **по CV** (5 фолдов, единый
протокол), holdout добавляется только финальной точке. Это эпистемически корректно:
holdout — финальная демонстрация, а не инструмент сравнения вариантов.

**CP coverage — per-target, не joint.** Marginal coverage при α = 0.10 гарантировано
**отдельно** для каждого из `y1, y2, y3`. Joint coverage по тройке (вероятность, что
**все три** интервала покрыли свои таргеты одновременно) не контролируется. Из
union-bound: joint coverage ≥ 1 − 3α = 0.70, что слабее. Bonferroni-коррекция
(α/3 = 0.033) сделала бы интервалы шире на ~30%. Для защиты курсовой текущий вариант
приемлем, но это явно зафиксировано в docstring `core/conformal.py`.

**Один сид для ablation Set Transformer.** Разрыв 1.4 п.п. между вариантами A (full
attention) и B (mean-pool) находится в пределах std прогонов — формально не значим.
Multi-seed прогоны на Colab укрепили бы вывод; на CPU они невозможны (3 ч × 5 сидов).

**Monotonicity = 0% у MC через post-hoc сортировку.** У Set Transformer 0% — by
construction (softmax + tail-cumsum математически гарантируют). У MC — операционно:
после получения трёх предсказаний `(p1, p2, p3)` применяется
`np.sort(...)[::-1]`. Оба отчитываются как «0%», но свойство у MC слабее (если оценщик
выдал `p1 < p2`, факт нарушения скрыт сортировкой; у Set Transformer такой ситуации
просто не может возникнуть).

**Plавающая параметризация β-CatBoost.** Поправочный CatBoost для MC обучается на
лог-отношении `log((y_true + ε) / (y_naive + ε))`. Зависит от ε и от выбора 200
деревьев. Сейчас гиперпараметры β-CatBoost не оптимизировались Optuna — это потенциальное
улучшение.

**Bibliography.** В `thesis/bibliography.bib` присутствуют заглушки
`author = {Anonymous}` в нескольких записях. Файл живёт в `thesis/`, которое не тронуто
в рефакторинге; необходимо привести в порядок до защиты.

**`catboost-log_link` и `catboost-smlar_smooth` не сагрегированы.** Оба варианта
посчитаны в `notebooks/method_1_catboost/02_train_default_and_optuna.ipynb` и
зафиксированы в `results/per_target_summary.json` (`catboost-log_link` только;
`catboost-smlar_smooth` отброшен как fail). Но `results/per_method/catboost.json`
и `results/final_comparison.csv` всё ещё содержат только default+optuna — при
следующем прогоне `core/runner.py` агрегатор нужно расширить, чтобы log-link попал в
сводную таблицу, а не только в README/REPORT.

---

## 9. Как запускать

```bash
# Установка зависимостей
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Прогон методов через ноутбуки (CatBoost ~5 мин, MLP ~2 мин, MC ~6 мин)
jupyter notebook notebooks/method_1_catboost/      # → results/per_method/catboost.{json,md}
jupyter notebook notebooks/method_2_mlp/           # → results/per_method/mlp.{json,md}
jupyter notebook notebooks/method_3_monte_carlo/   # → results/per_method/monte_carlo.{json,md}

# Set Transformer: ablation обучается на Colab GPU (5 фолдов × 18.8 мин),
# сборка локально из заморожённого CSV (~5 с):
python notebooks/method_4_set_transformer/build_results.py
# Holdout для A_full локально из весов через 03_evaluation_and_results.ipynb (~3 мин CPU).

# Агрегация в итоговую таблицу: results/final_comparison.{csv,md} + per_target_summary.json
python -m core.runner

# Проверка тестов (33 теста, все должны проходить)
pytest tests/

# Демонстрационный Streamlit-дашборд (3 вкладки: О проекте / Методы / Сравнение)
streamlit run streamlit_app/app.py
```

**Воспроизводимость.** Все сиды зашиты в `core/config.py:SEED = 42`. Сторонняя
рандомизация (Optuna TPE, PyTorch init, MC sampling) использует тот же сид через
`np.random.default_rng(SEED)` или `torch.manual_seed(SEED)`. При корректной установке
зависимостей из `requirements.txt` все цифры в таблице §5 воспроизводимы до знака после
запятой.
