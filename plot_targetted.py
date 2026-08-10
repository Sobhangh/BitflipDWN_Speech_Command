import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


param_DWN_2_4000 = (2**2) * 4000 + (2**2) * 4000 * 0.5
param_DWN_4_4000 = (2**4) * 4000 + (2**4) * 4000 * 0.5
param_DWN_6_4000 = (2**6) * 4000 + (2**6) * 4000 * 0.5
param_DWN_2_16000 = (2**2) * 16000 + (2**2) * 16000 * 0.5
param_DWN_6_1000 = (2**6) * 1000 + (2**6) * 1000 * 0.5

param_ConvBNN_5 = 1319453
param_BNN_100 = 89810

PARAM_MAP = {
	"DWN_2_4000": int(param_DWN_2_4000),
	"DWN_4_4000": int(param_DWN_4_4000),
	"DWN_6_4000": int(param_DWN_6_4000),
	"DWN_2_16000": int(param_DWN_2_16000),
	"DWN_6_1000": int(param_DWN_6_1000),
	"ConvBNN_5": int(param_ConvBNN_5),
	"BNN_100": int(param_BNN_100),
}


def parse_targeted_filename(file_path: Path) -> dict[str, str | int | None]:
	stem_parts = file_path.stem.split("_")
	# Expected:
	# bit_flip_targeted_<task>_<lut_rank>_<layer_size>
	# bit_flip_targeted_(Conv)BNN_<task>_<layer_size>
	if len(stem_parts) < 6 or stem_parts[:3] != ["bit", "flip", "targeted"]:
		raise ValueError(f"Unexpected file name format: {file_path.name}")

	marker = stem_parts[3]
	if marker in {"BNN", "ConvBNN"}:
		task = stem_parts[4]
		layer_size = int(stem_parts[5])
		model_type = marker
		param_key = f"{marker}_{layer_size}"
		lut_rank = None
	else:
		task = marker
		lut_rank = int(stem_parts[4])
		layer_size = int(stem_parts[5])
		model_type = f"DWN (lut_rank={lut_rank})"
		param_key = f"DWN_{lut_rank}_{layer_size}"

	return {
		"task": task,
		"model_type": model_type,
		"lut_rank": lut_rank,
		"layer_size": layer_size,
		"param_key": param_key,
	}


def _safe_float(value: str) -> float | None:
	value = value.strip()
	if value == "":
		return None
	return float(value)


def collect_level_stats(file_path: Path, model_type: str) -> tuple[float, dict[int, dict[tuple[int, int], dict[str, list[float]]]]]:
	rows: list[dict[str, float | int | None]] = []

	with file_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(
				{
					"source": int(row["source"]),
					"target": int(row["target_class"]),
					"level": int(row["level"]),
					"base_acc": float(row["base_acc"]),
					"nb_flips": float(row["nb_flips"]),
					"last_asr": float(row["last_asr"]),
					"last_ta": _safe_float(row["last_ta"]),
				}
			)

	if not rows:
		raise ValueError(f"No rows in file: {file_path.name}")

	base_acc = mean([float(r["base_acc"]) for r in rows])
	is_dwn = model_type.startswith("DWN")

	level_bucket: dict[int, dict[tuple[int, int], dict[str, list[float]]]] = {}

	for row in rows:
		row_asr = float(row["last_asr"])
		if is_dwn and row_asr < base_acc:
			continue

		level = int(row["level"])
		key = (int(row["source"]), int(row["target"]))
		level_bucket.setdefault(level, {}).setdefault(
			key,
			{"nb_flips": [], "last_asr": [], "last_ta": []},
		)

		level_bucket[level][key]["nb_flips"].append(float(row["nb_flips"]))
		level_bucket[level][key]["last_asr"].append(row_asr)
		if row["last_ta"] is not None:
			level_bucket[level][key]["last_ta"].append(float(row["last_ta"]))

	return base_acc, level_bucket


def aggregate_pair_metrics(pair_data: dict[str, list[float]]) -> dict[str, float]:
	nb_vals = pair_data["nb_flips"]
	asr_vals = pair_data["last_asr"]
	ta_vals = pair_data["last_ta"]

	return {
		"nb_mean": mean(nb_vals) if nb_vals else np.nan,
		"nb_std": stdev(nb_vals) if len(nb_vals) > 1 else 0.0,
		"asr_mean": mean(asr_vals) if asr_vals else np.nan,
		"ta_mean": mean(ta_vals) if ta_vals else np.nan,
	}


def save_level1_table(
	stem: str,
	task: str,
	model_type: str,
	parameters: int,
	base_acc: float,
	level_data: dict[tuple[int, int], dict[str, list[float]]],
	report_dir: Path,
) -> None:
	nb_row: list[str] = []
	asr_row: list[str] = []

	for target in range(10):
		pair = level_data.get((-1, target))
		if pair is None:
			nb_row.append("N/A")
			asr_row.append("N/A")
			continue

		agg = aggregate_pair_metrics(pair)
		nb_mean = agg["nb_mean"]
		nb_std = agg["nb_std"]
		asr_mean = agg["asr_mean"]

		if np.isnan(nb_mean):
			nb_row.append("N/A")
		else:
			nb_row.append(f"{nb_mean:.1f} +/- {nb_std:.1f}")

		if np.isnan(asr_mean):
			asr_row.append("N/A")
		else:
			asr_row.append(f"{asr_mean:.4f}")

	fig, ax = plt.subplots(figsize=(15, 3.8))
	ax.axis("off")

	col_labels = [str(i) for i in range(10)]
	row_labels = ["nb_flips mean +/- std", "mean last_asr"]
	table_data = [nb_row, asr_row]

	table = ax.table(
		cellText=table_data,
		rowLabels=row_labels,
		colLabels=col_labels,
		loc="center",
		cellLoc="center",
		colLoc="center",
	)
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.scale(1, 1.6)

	for (row_idx, col_idx), cell in table.get_celld().items():
		cell.set_edgecolor("#C3CBD7")
		if row_idx == 0:
			cell.set_facecolor("#E6EEF8")
			cell.set_text_props(weight="bold")
		elif col_idx == -1:
			cell.set_facecolor("#F3F6FB")
			cell.set_text_props(weight="bold")
		else:
			cell.set_facecolor("#FFFFFF")

	title = (
		f"Targeted Level 1 Summary | task={task} | model={model_type} | "
		f"params={parameters:,} | base_acc={base_acc:.4f}"
	)
	ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

	output_file = report_dir / f"targeted_{stem}_level1_table.png"
	fig.tight_layout()
	fig.savefig(output_file, dpi=220, bbox_inches="tight")
	plt.close(fig)


def _draw_single_heatmap(
	ax: plt.Axes,
	matrix: np.ndarray,
	title: str,
	fmt: str,
	std_matrix: np.ndarray | None,
	red_mask: np.ndarray,
) -> None:
	cmap = plt.cm.Blues.copy()
	cmap.set_bad("#EDEDED")
	im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect="equal")

	ax.set_title(title, fontsize=11, fontweight="bold")
	ax.set_xlabel("Target class")
	ax.set_ylabel("Source class")
	ax.set_xticks(range(10))
	ax.set_yticks(range(10))

	ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
	ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
	ax.grid(which="minor", color="white", linewidth=1.2)
	ax.tick_params(which="minor", bottom=False, left=False)

	for src in range(10):
		for tgt in range(10):
			value = matrix[src, tgt]

			if red_mask[src, tgt]:
				ax.add_patch(
					plt.Rectangle(
						(tgt - 0.5, src - 0.5),
						1,
						1,
						facecolor=(1, 0, 0, 0.28),
						edgecolor="none",
					)
				)

			if np.isnan(value):
				label = "x" if src == tgt else "-"
				ax.text(tgt, src, label, ha="center", va="center", fontsize=9, color="#5B6573")
				continue

			if std_matrix is not None:
				std_value = std_matrix[src, tgt]
				label = f"{value:.1f}\n+/-{std_value:.1f}"
				fontsize = 8.2
			else:
				label = format(value, fmt)
				fontsize = 8.8

			ax.text(tgt, src, label, ha="center", va="center", fontsize=fontsize, color="black")

	plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def save_level23_plots(
	stem: str,
	task: str,
	model_type: str,
	parameters: int,
	base_acc: float,
	level: int,
	level_data: dict[tuple[int, int], dict[str, list[float]]],
	report_dir: Path,
) -> None:
	nb_mean = np.full((10, 10), np.nan)
	nb_std = np.full((10, 10), np.nan)
	asr_mean = np.full((10, 10), np.nan)
	ta_mean = np.full((10, 10), np.nan)

	for (source, target), pair_data in level_data.items():
		if source < 0 or source > 9 or target < 0 or target > 9:
			continue

		agg = aggregate_pair_metrics(pair_data)
		nb_mean[source, target] = agg["nb_mean"]
		nb_std[source, target] = agg["nb_std"]
		asr_mean[source, target] = agg["asr_mean"]
		ta_mean[source, target] = agg["ta_mean"]

	red_mask = np.zeros((10, 10), dtype=bool)

	fig, axes = plt.subplots(1, 3, figsize=(23, 7.2), constrained_layout=True)
	fig.suptitle(
		(
			f"Targeted Level {level} | task={task} | model={model_type} | "
			f"params={parameters:,} | base_acc={base_acc:.4f}"
		),
		fontsize=13,
		fontweight="bold",
	)

	_draw_single_heatmap(
		axes[0],
		nb_mean,
		"Mean nb_flips with std",
		".1f",
		std_matrix=nb_std,
		red_mask=red_mask,
	)
	_draw_single_heatmap(
		axes[1],
		asr_mean,
		"Mean last_asr",
		".4f",
		std_matrix=None,
		red_mask=red_mask,
	)
	_draw_single_heatmap(
		axes[2],
		ta_mean,
		"Mean last_ta",
		".4f",
		std_matrix=None,
		red_mask=red_mask,
	)

	output_file = report_dir / f"targeted_{stem}_level{level}_plots.png"
	fig.savefig(output_file, dpi=220, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	project_root = Path(__file__).resolve().parent
	results_dir = project_root / "results"
	report_dir = project_root / "report"
	report_dir.mkdir(parents=True, exist_ok=True)

	files = sorted(results_dir.glob("bit_flip_targeted_*.csv"))
	if not files:
		print(f"No targeted result files found in: {results_dir}")
		return

	missing_param_keys: set[str] = set()

	for csv_file in files:
		parsed = parse_targeted_filename(csv_file)
		task = str(parsed["task"])
		model_type = str(parsed["model_type"])
		param_key = str(parsed["param_key"])
		parameters = PARAM_MAP.get(param_key)

		if parameters is None:
			missing_param_keys.add(param_key)
			continue

		base_acc, level_bucket = collect_level_stats(csv_file, model_type)
		stem = csv_file.stem

		if 1 in level_bucket:
			save_level1_table(
				stem=stem,
				task=task,
				model_type=model_type,
				parameters=int(parameters),
				base_acc=base_acc,
				level_data=level_bucket[1],
				report_dir=report_dir,
			)
			print(f"Saved image: {report_dir / f'targeted_{stem}_level1_table.png'}")

		for level in (2, 3):
			if level not in level_bucket:
				continue
			save_level23_plots(
				stem=stem,
				task=task,
				model_type=model_type,
				parameters=int(parameters),
				base_acc=base_acc,
				level=level,
				level_data=level_bucket[level],
				report_dir=report_dir,
			)
			print(f"Saved image: {report_dir / f'targeted_{stem}_level{level}_plots.png'}")

	if missing_param_keys:
		print("\nMissing parameter mappings for:")
		for key in sorted(missing_param_keys):
			print(f"- {key}")


if __name__ == "__main__":
	main()
