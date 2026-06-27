# 36 — 全33業種 因果レジーム検知バッチ結果

> 自動生成: `causal_sector_pipeline.py --all-sectors --fast`
> 成功: 33 業種

## 1. ATE(VALUE→RET) 分布

- 正（バリュー効果あり）: 21 業種
- 負（逆/弱い）: 12 業種

## 2. 業種種別サマリー

| kind | n | mean ATE | mean 因果警報数 |
|---|---:|---:|---:|
| default | 18 | +0.001516 | 10.0 |
| energy_util | 2 | +0.001902 | 11.5 |
| financial | 4 | -0.001386 | 13.0 |
| heavy_industry | 6 | +0.001911 | 9.8 |
| it_comm | 3 | -0.000260 | 12.3 |

## 3. 全業種一覧

| S33 | 業種 | kind | ATE | 因果警報 | Collider |
|---|---|---|---:|---:|---|
| 0050 | 水産・農林業 | default | +0.0015 | 8 | MOM, SHORT_RATIO, VOL |
| 1050 | 鉱業 | default | -0.0008 | 8 | MOM, SHORT_RATIO, VOL |
| 2050 | 建設業 | default | -0.0002 | 14 | MOM, SHORT_RATIO, VOL |
| 3050 | 食料品 | default | +0.0021 | 15 | MOM, SHORT_RATIO, VOL |
| 3100 | 繊維製品 | default | -0.0001 | 10 | MOM, SHORT_RATIO, VOL |
| 3150 | パルプ・紙 | default | +0.0020 | 13 | MOM, SHORT_RATIO, VOL |
| 3200 | 化学 | default | +0.0024 | 12 | MOM, SHORT_RATIO, VOL |
| 3250 | 医薬品 | default | +0.0050 | 8 | MOM, SHORT_RATIO, VOL |
| 3300 | 石油･石炭製品 | energy_util | +0.0015 | 8 | SHORT_RATIO, VOL |
| 3350 | ゴム製品 | default | +0.0035 | 6 | MOM, SHORT_RATIO, VOL |
| 3400 | ガラス･土石製品 | heavy_industry | +0.0021 | 9 | MOM, SHORT_RATIO, VOL |
| 3450 | 鉄鋼 | heavy_industry | +0.0047 | 9 | MOM, SHORT_RATIO, VOL |
| 3500 | 非鉄金属 | heavy_industry | -0.0008 | 10 | MOM, SHORT_RATIO, VOL |
| 3550 | 金属製品 | heavy_industry | +0.0027 | 8 | MOM, SHORT_RATIO, VOL |
| 3600 | 機械 | heavy_industry | -0.0005 | 9 | MOM, SHORT_RATIO, VOL |
| 3650 | 電気機器 | it_comm | +0.0058 | 12 | MOM, SHORT_RATIO, VOL |
| 3700 | 輸送用機器 | heavy_industry | +0.0032 | 14 | MOM, SHORT_RATIO, VOL |
| 3750 | 精密機器 | it_comm | +0.0008 | 13 | MOM, SHORT_RATIO, VOL |
| 3800 | その他製品 | default | +0.0008 | 8 | MOM, SHORT_RATIO, VOL |
| 4050 | 電気･ガス業 | energy_util | +0.0023 | 15 | SHORT_RATIO, VOL |
| 5050 | 陸運業 | default | +0.0017 | 15 | MOM, SHORT_RATIO, VOL |
| 5100 | 海運業 | default | -0.0008 | 13 | MOM, SHORT_RATIO, VOL |
| 5150 | 空運業 | default | -0.0011 | 2 | MOM, SHORT_RATIO, VOL |
| 5200 | 倉庫･運輸関連業 | default | -0.0026 | 11 | MOM, SHORT_RATIO, VOL |
| 5250 | 情報･通信業 | it_comm | -0.0074 | 12 | MOM, SHORT_RATIO, VOL |
| 6050 | 卸売業 | default | -0.0032 | 10 | MOM, SHORT_RATIO, VOL |
| 6100 | 小売業 | default | +0.0053 | 8 | MOM, SHORT_RATIO, VOL |
| 7050 | 銀行業 | financial | -0.0089 | 10 | MOM, SHORT_RATIO, VOL |
| 7100 | 証券･商品先物取引業 | financial | +0.0018 | 11 | MOM, SHORT_RATIO, VOL |
| 7150 | 保険業 | financial | -0.0010 | 15 | MOM, SHORT_RATIO, VOL |
| 7200 | その他金融業 | financial | +0.0026 | 16 | MOM, SHORT_RATIO, VOL |
| 8050 | 不動産業 | default | +0.0116 | 9 | MOM, SHORT_RATIO, VOL |
| 9050 | サービス業 | default | +0.0001 | 10 | MOM, SHORT_RATIO, VOL |

## 4. 構造変化が多い業種（因果警報上位5）

- **7200 その他金融業**: 16件  ATE=+0.0026
- **3050 食料品**: 15件  ATE=+0.0021
- **4050 電気･ガス業**: 15件  ATE=+0.0023
- **5050 陸運業**: 15件  ATE=+0.0017
- **7150 保険業**: 15件  ATE=-0.0010