# Action Smoothing Benchmark

這是一個與 LeRobot 原始碼完全分離的離線測試工具。它從 LeRobot v3 dataset 的 parquet 讀取逐幀 `action`，重建固定長度 action chunks，並比較多種平滑方法。

## 重要限制

指定資料集只保存逐幀 `action`，沒有 policy 原始輸出 chunk 或網路傳輸邊界。本工具預設按 episode 每 50 幀做非重疊切分，因此結果是部署流程的近似測試，不是原始 action chunks 的逐塊重播。

## 方法

- 未處理的 raw baseline
- 1 至 5 次全 chunk 多項式最小平方擬合
- 因果 EMA
- 因果與 zero-phase Butterworth 低通濾波
- Savitzky-Golay
- Gaussian filter
- 以 GCV 選擇平滑強度的 cubic smoothing spline

報告會分開標示因果方法與必須先取得完整 chunk 的方法。現有 LeRobot 三次式對應 `polynomial_3`。

## 安裝與執行

```powershell
cd C:\Users\ccu\mujoco_ur5_graph\action_smoothing_benchmark
$env:UV_CACHE_DIR='.uv-cache'
uv sync --extra test
uv run action-smoothing-benchmark
```

預設資料來源已設為：

```text
C:\Users\ccu\.cache\huggingface\lerobot\wuc1\rollout_bi_so101_ffp_0615-14-12-dagger_merged3cam_nopolicyaction_no_use_smoothing
```

指定其他資料或參數：

```powershell
uv run action-smoothing-benchmark `
  --dataset-root C:\path\to\dataset `
  --chunk-length 50 `
  --output-dir results
```

## 輸出

- `results_smoothing_first/report.html`：互動式排名、Pareto 圖與代表性 chunk 曲線。
- `results_smoothing_first/summary.csv`：每個方法與參數組合的彙總指標及部署折衷分數。
- `results_smoothing_first/per_chunk.csv`：逐 episode、chunk、關節的完整指標。
- `results_smoothing_first/figures/*.png`：raw、現有 cubic 與最高分方法的靜態比較圖。
- `results_smoothing_first/run_config.json`：資料、環境、方法參數與評分權重。
- `results_smoothing_first/errors.json`：短 chunk fallback 或數值錯誤。

Deployment score is based on smoothness 60%, high-frequency energy 20%, and runtime 20%. RMSE only measures deviation from the raw action and is diagnostic, not a quality ground truth. Low-frequency ratio is diagnostic only because it complements the high-frequency ratio after removing DC. Seam metrics are omitted. A single weighted score cannot replace real-robot validation; also inspect raw metrics, causality, and joint limits.

## 測試

```powershell
uv run pytest
```

