from __future__ import annotations

import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import ActionChunk, DatasetInfo
from .metrics import BenchmarkResult


def select_representative_chunks(chunks: list[ActionChunk], fps: float) -> dict[str, ActionChunk]:
    jerk = []
    for chunk in chunks:
        values = np.diff(chunk.action, n=3, axis=0) * fps**3
        jerk.append(float(np.sqrt(np.mean(values**2))) if values.size else 0.0)
    order = np.argsort(jerk)
    selected = {"median_jerk": chunks[int(order[len(order) // 2])], "highest_jerk": chunks[int(order[-1])]}
    intervention = [chunk for chunk in chunks if np.any(chunk.intervention)]
    if intervention:
        selected["intervention"] = max(intervention, key=lambda chunk: float(np.mean(chunk.intervention)))
    return selected


def _curve_figure(chunk: ActionChunk, result: BenchmarkResult, action_names: list[str], method_ids: list[str], label: str) -> go.Figure:
    figure = make_subplots(rows=4, cols=3, subplot_titles=action_names, shared_xaxes=True)
    time = np.arange(len(chunk.action))
    for method_id in method_ids:
        output = result.outputs[(chunk.episode_index, chunk.chunk_index, method_id)]
        for dimension, _name in enumerate(action_names):
            row, col = divmod(dimension, 3)
            figure.add_trace(go.Scatter(x=time, y=output[:, dimension], mode="lines", name=method_id, legendgroup=method_id, showlegend=dimension == 0), row=row + 1, col=col + 1)
    figure.update_layout(height=920, title=f"{label}: episode {chunk.episode_index}, chunk {chunk.chunk_index}", margin=dict(l=40, r=20, t=70, b=40))
    figure.update_xaxes(title_text="step")
    return figure


def _write_static_figures(output_dir: Path, selected: dict[str, ActionChunk], result: BenchmarkResult, action_names: list[str], best_method: str) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    method_ids = list(dict.fromkeys(["raw", "polynomial_3", best_method]))
    for label, chunk in selected.items():
        figure, axes = plt.subplots(4, 3, figsize=(16, 13), sharex=True, constrained_layout=True)
        for dimension, axis in enumerate(axes.flat):
            for method_id in method_ids:
                output = result.outputs[(chunk.episode_index, chunk.chunk_index, method_id)]
                axis.plot(output[:, dimension], label=method_id, linewidth=1.4)
            axis.set_title(action_names[dimension], fontsize=9)
            axis.grid(alpha=0.25)
        axes.flat[0].legend(fontsize=8)
        figure.suptitle(f"{label} | episode {chunk.episode_index}, chunk {chunk.chunk_index}")
        figure.savefig(figure_dir / f"{label}.png", dpi=160)
        plt.close(figure)


def write_report(output_dir: Path, info: DatasetInfo, chunks: list[ActionChunk], result: BenchmarkResult, warnings: list[str]) -> None:
    selected = select_representative_chunks(chunks, info.fps)
    ranked = result.summary[result.summary["method_id"] != "raw"]
    best_method = str(ranked.iloc[0]["method_id"])
    chosen_methods = list(dict.fromkeys(["raw", "polynomial_3", *ranked.head(5)["method_id"].tolist()]))
    pareto = px.scatter(
        result.summary, x="rmse", y="jerk_rms", color="causal", size="runtime_p95_ms",
        hover_name="method_id", hover_data=["deployment_score", "low_frequency_ratio", "high_frequency_ratio"],
        title="原始偏離與 jerk 的診斷（僅供參考）",
    )
    ranking_columns = ["rank", "method_id", "causal", "deployment_score", "rmse", "jerk_rms", "acceleration_rms", "low_frequency_ratio", "high_frequency_ratio", "runtime_p95_ms"]
    ranking_html = result.summary[ranking_columns].round(5).to_html(index=False, classes="ranking", border=0)
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in warnings) or "<li>無資料完整性警告</li>"
    figure_html = [pareto.to_html(full_html=False, include_plotlyjs=True)]
    for label, chunk in selected.items():
        figure_html.append(_curve_figure(chunk, result, info.action_names, chosen_methods, label).to_html(full_html=False, include_plotlyjs=False))
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Action smoothing benchmark</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:0;color:#1d232a;background:#f5f6f8}}main{{max-width:1500px;margin:auto;padding:24px}}h1{{font-size:28px}}section{{background:white;border:1px solid #d9dde3;margin:16px 0;padding:18px;border-radius:6px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px 9px;border-bottom:1px solid #e4e7eb;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}.facts{{display:flex;gap:28px;flex-wrap:wrap}}.facts b{{display:block;font-size:21px}}</style></head>
<body><main><h1>Action Chunk 平滑方法比較</h1>
<section><div class="facts"><span><b>{info.total_frames}</b>frames</span><span><b>{info.total_episodes}</b>episodes</span><span><b>{len(chunks)}</b>reconstructed chunks</span><span><b>{info.fps:g} Hz</b>sampling rate</span><span><b>{best_method}</b>highest deployment score</span></div>
<p>Scoring uses smoothness 60%, high-frequency energy 20%, and runtime 20%. RMSE and low-frequency ratio are diagnostic only. Seam metrics are omitted.</p><ul>{warning_html}</ul></section>
<section><h2>總排名</h2>{ranking_html}</section>
<section>{figure_html[0]}</section>
{''.join(f'<section>{item}</section>' for item in figure_html[1:])}
</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")
    _write_static_figures(output_dir, selected, result, info.action_names, best_method)
