import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


param_DWN_2_4000 = (2**2) * 4000 + (2**2) * 4000 * 0.5
param_DWN_4_4000 = (2**4) * 4000 + (2**4) * 4000 * 0.5
param_DWN_6_4000 = (2**6) * 4000 + (2**6) * 4000 * 0.5
param_DWN_2_16000 = (2**2) * 16000 + (2**2) * 16000 * 0.5
param_DWN_6_1000 = (2**6) * 1000 + (2**6) * 1000 * 0.5

param_ConvBNN_5 = 1_319_453
param_ConvBNN_64 = 17_338_307
param_BNN_100 = 89_810
param_BNN_100_6 = 120_410

PARAM_MAP = {
	"DWN_2_4000": int(param_DWN_2_4000),
	"DWN_4_4000": int(param_DWN_4_4000),
	"DWN_6_4000": int(param_DWN_6_4000),
	"DWN_2_16000": int(param_DWN_2_16000),
	"DWN_6_1000": int(param_DWN_6_1000),
	"ConvBNN_5": int(param_ConvBNN_5),
	"ConvBNN_64": int(param_ConvBNN_64),
	"BNN_100": int(param_BNN_100),
	"BNN_100_6": int(param_BNN_100_6),
}


def parse_targeted_filename(file_path: Path) -> dict[str, str | int | None]:
	stem_parts = file_path.stem.split("_")
	if len(stem_parts) < 6 or stem_parts[:3] != ["bit", "flip", "targeted"]:
		raise ValueError(f"Unexpected file name format: {file_path.name}")

	marker = stem_parts[3]
	if marker in {"BNN", "ConvBNN"}:
		task = stem_parts[4]
		layer_tokens = stem_parts[5:]
		layer_size = int(layer_tokens[0])
		if marker == "BNN":
			model_type = "BNN (3 layers)" if len(layer_tokens) == 1 else f"BNN ({layer_tokens[-1]} layers)"
		else:
			model_type = f"ConvBNN ({layer_size})"
		param_key = f"{marker}_{'_'.join(layer_tokens)}"
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


def read_targeted_rows(file_path: Path) -> list[dict[str, float | int | None]]:
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
	return rows


def _plot_hist(
	ax: plt.Axes,
	data: list[float],
	title: str,
	x_label: str,
	color: str,
	bins: int | list[float] | str,
) -> None:
	if not data:
		ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=11)
		ax.set_title(title, fontsize=11, fontweight="bold")
		ax.set_xlabel(x_label)
		ax.set_ylabel("Count")
		return

	ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
	ax.set_title(title, fontsize=11, fontweight="bold")
	ax.set_xlabel(x_label)
	ax.set_ylabel("Count")
	ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)


def save_level_histograms(
	stem: str,
	task: str,
	model_type: str,
	parameters: int,
	level: int,
	rows: list[dict[str, float | int | None]],
	report_dir: Path,
) -> None:
	level_rows = [row for row in rows if int(row["level"]) == level]
	if not level_rows:
		return

	base_acc = mean([float(row["base_acc"]) for row in level_rows])
	plot_rows = level_rows
	# if model_type.startswith("DWN"):
	# 	# Keep behavior consistent with plot_targetted for DWN summaries.
	# 	plot_rows = [row for row in level_rows if float(row["last_asr"]) >= base_acc]

	total_trials = len(level_rows)
	success_trials = sum(1 for row in level_rows if float(row["last_asr"]) >= base_acc)
	success_rate_pct = 100.0 * success_trials / total_trials if total_trials else 0.0

	nb_flips_vals = [float(row["nb_flips"]) for row in plot_rows]
	asr_vals = [float(row["last_asr"]) for row in plot_rows]
	ta_vals = [float(row["last_ta"]) for row in plot_rows if row["last_ta"] is not None]

	if level == 2:
		fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), constrained_layout=True)
	else:
		fig, axes = plt.subplots(1, 3, figsize=(21, 5.3), constrained_layout=True)

	fig.suptitle(
		(
			f"Targeted Level {level} Histograms | task={task} | model={model_type} | "
			f"params={parameters:,} | base_acc={base_acc:.4f}"
		),
		fontsize=13,
		fontweight="bold",
	)

	_plot_hist(
		axes[0],
		nb_flips_vals,
		"nb_flips",
		"Number of bit flips",
		"#2E8B57",
		bins="auto",
	)

	_plot_hist(
		axes[1],
		asr_vals,
		"last_asr",
		"ASR",
		"#4C78A8",
		bins=[i / 100 for i in range(0, 101, 2)],
	)
	baseline_label = (
		f"Baseline={base_acc:.4f} | successful attacks: {success_rate_pct:.1f}%"
	)
	axes[1].axvline(base_acc, color="#C62828", linestyle="--", linewidth=2, label=baseline_label)
	axes[1].legend(loc="best", fontsize=9)

	if level == 3:
		_plot_hist(
			axes[2],
			ta_vals,
			"last_ta",
			"Post-attack test accuracy",
			"#F58518",
			bins=[i / 100 for i in range(0, 101, 2)],
		)

	output_file = report_dir / f"targeted_hist_{stem}_level{level}.png"
	fig.savefig(output_file, dpi=220, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	project_root = Path(__file__).resolve().parent
	results_dir = project_root / "results"
	report_dir = project_root / "report" / "histograms"
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

		rows = read_targeted_rows(csv_file)
		stem = csv_file.stem

		for level in (2, 3):
			save_level_histograms(
				stem=stem,
				task=task,
				model_type=model_type,
				parameters=int(parameters),
				level=level,
				rows=rows,
				report_dir=report_dir,
			)
			print(f"Saved image: {report_dir / f'targeted_hist_{stem}_level{level}.png'}")

	if missing_param_keys:
		print("\nMissing parameter mappings for:")
		for key in sorted(missing_param_keys):
			print(f"- {key}")


if __name__ == "__main__":
	main()
