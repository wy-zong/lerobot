# `bi_so101_ffp_0615-14-12-dagger_merged` 合併歷史

記錄日期：2026-06-23

## 目標資料集

- Repo ID：`wuc1/bi_so101_ffp_0615-14-12-dagger_merged`
- 本機路徑：`/home/wy/.cache/huggingface/lerobot/wuc1/bi_so101_ffp_0615-14-12-dagger_merged`
- Episodes：248
- Frames：812,393

## 完整合併樹

```text
wuc1/bi_so101_ffp_0615-14-12-dagger_merged
248 episodes / 812,393 frames
│
├─ wuc1/bi_so101_ffp_0615-14-12_merged_with_intervention
│  198 episodes / 583,727 frames
│  │
│  └─ wuc1/bi_so101_ffp_0615-14-12_merged
│     ├─ wuc1/bi_so101_ffp_20260615_merged
│     │  ├─ wuc1/bi_so101_ffp_20260615_020219
│     │  │  25 episodes / 82,214 frames
│     │  └─ wuc1/bi_so101_ffp_20260615_001830
│     │     50 episodes / 118,120 frames
│     │
│     ├─ wuc1/bi_so101_ffp_20260614_merged
│     │  ├─ wuc1/bi_so101_ffp_20260614_175107
│     │  │  16 episodes / 42,038 frames
│     │  └─ wuc1/bi_so101_ffp_20260614_202633
│     │     7 episodes / 26,404 frames
│     │
│     └─ wuc1/bi_so101_ffp_0612_merged
│        ├─ wuc1/bi_so101_ffp_20260603_200349
│        │  50 episodes / 126,746 frames
│        └─ wuc1/bi_so101_ffp_free_style_20260610_190734
│           50 episodes / 188,205 frames
│
└─ wuc1/rollout_dagger_bi_so101_ffp_0615-14-12_merged
   50 episodes / 228,666 frames
```

## 最底層來源資料集

| 合併順序 | Repo ID | Episodes | Frames | 最終 episode 範圍 |
|---:|---|---:|---:|---:|
| 1 | `wuc1/bi_so101_ffp_20260615_020219` | 25 | 82,214 | 0–24 |
| 2 | `wuc1/bi_so101_ffp_20260615_001830` | 50 | 118,120 | 25–74 |
| 3 | `wuc1/bi_so101_ffp_20260614_175107` | 16 | 42,038 | 75–90 |
| 4 | `wuc1/bi_so101_ffp_20260614_202633` | 7 | 26,404 | 91–97 |
| 5 | `wuc1/bi_so101_ffp_20260603_200349` | 50 | 126,746 | 98–147 |
| 6 | `wuc1/bi_so101_ffp_free_style_20260610_190734` | 50 | 188,205 | 148–197 |
| 7 | `wuc1/rollout_dagger_bi_so101_ffp_0615-14-12_merged` | 50 | 228,666 | 198–247 |
| **總計** |  | **248** | **812,393** | **0–247** |

## 合併階段

### 1. 2026-06-15 資料

`wuc1/bi_so101_ffp_20260615_merged`：

```text
20260615_020219（25 episodes / 82,214 frames）
+ 20260615_001830（50 episodes / 118,120 frames）
= 75 episodes / 200,334 frames
```

Episode metadata 顯示實際順序是 `020219` 在前、`001830` 在後。

### 2. 2026-06-14 資料

`wuc1/bi_so101_ffp_20260614_merged`：

```text
20260614_175107（16 episodes / 42,038 frames）
+ 20260614_202633（7 episodes / 26,404 frames）
= 23 episodes / 68,442 frames
```

### 3. 2026-06-12 資料

`wuc1/bi_so101_ffp_0612_merged`：

```text
20260603_200349（50 episodes / 126,746 frames）
+ free_style_20260610_190734（50 episodes / 188,205 frames）
= 100 episodes / 314,951 frames
```

### 4. 合併 6/15、6/14、6/12

`wuc1/bi_so101_ffp_0615-14-12_merged`：

```text
20260615_merged（75 episodes / 200,334 frames）
+ 20260614_merged（23 episodes / 68,442 frames）
+ 0612_merged（100 episodes / 314,951 frames）
= 198 episodes / 583,727 frames
```

### 5. 加入 DAgger `intervention` 欄位

腳本 `/home/wy/好用的指令/資料集轉換成dagger格式.py` 將：

```text
wuc1/bi_so101_ffp_0615-14-12_merged
```

轉換成：

```text
wuc1/bi_so101_ffp_0615-14-12_merged_with_intervention
```

這個步驟沒有增加或刪除 episodes。它為全部 583,727 frames 加入值為
`True` 的 `intervention` 欄位。

### 6. 最終合併

腳本 `/home/wy/好用的指令/merge.sh` 將：

```text
0615-14-12_merged_with_intervention（198 episodes / 583,727 frames）
+ rollout_dagger_bi_so101_ffp_0615-14-12_merged（50 episodes / 228,666 frames）
= 248 episodes / 812,393 frames
```

## 驗證方式

除了比對 `meta/info.json` 的 `total_episodes` 與 `total_frames`，也逐一讀取各資料集：

```text
meta/episodes/chunk-*/file-*.parquet
```

並比對每個 episode 的 `length` 序列。各階段的序列都能按上述順序精確串接成下一階段資料集；最終本機資料集也精確等於前 198 個示範 episodes 加上後 50 個 rollout episodes。

Frames 總和：

```text
82,214 + 118,120 + 42,038 + 26,404
+ 126,746 + 188,205 + 228,666
= 812,393
```

Episodes 總和：

```text
25 + 50 + 16 + 7 + 50 + 50 + 50 = 248
```

## 本機證據

- `/home/wy/好用的指令/merge.sh`
  - 記錄最終兩個直接輸入資料集。
- `/home/wy/好用的指令/資料集轉換成dagger格式.py`
  - 記錄 `intervention=True` 欄位的建立方式。
- `/home/wy/.vscode-server/data/User/History/14b16e43/8rh0.sh`
  - 記錄 `0615-14-12_merged` 的三個上游資料集。
- `/home/wy/合併lerobot_dataset.sh`
  - 記錄 `0612_merged` 的兩個上游資料集。

## 名稱相近但未合併的資料集

以下資料集不在這次合併結果中：

- `wuc1/bi_so101_ffp_20260614_202044`
  - 1 episode / 4,110 frames
- `wuc1/bi_so101_ffp_20260614_193013`
  - 1 episode / 752 frames
- `wuc1/bi_so101_ffp_20260603_200349_subtask`
  - 與 `20260603_200349` 具有相同的 50 個原始 episodes，但額外包含 subtask features；實際合併來源是沒有 `_subtask` 後綴的資料集。

## Rollout 資料集備註

`wuc1/rollout_dagger_bi_so101_ffp_0615-14-12_merged` 名稱包含 `_merged`，但目前沒有找到它由其他資料集執行 `lerobot-edit-dataset merge` 的紀錄。在這條資料 lineage 中，將它視為直接採集的 50-episode leaf dataset；名稱中的 `_merged` 應是引用執行 rollout 時所使用的上游模型或資料集名稱。

