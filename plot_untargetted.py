import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

#from bitflip_exp_bnn import MAX_NB_FLIP

param_DWN_2_4000 = (2**2) * 4000 + (2**2) * 4000 * 0.5
param_DWN_4_4000 = (2**4) * 4000 + (2**4) * 4000 * 0.5
param_DWN_6_4000 = (2**6) * 4000 + (2**6) * 4000 * 0.5
param_DWN_2_16000 = (2**2) * 16000 + (2**2) * 16000 * 0.5
param_DWN_6_1000 = (2**6) * 1000 + (2**6) * 1000 * 0.5
param_DWN_4_12000 = (2**4) * 12000 + (2**4) * 12000 * 0.5
param_DWN_6_12000 = (2**6) * 12000 + (2**6) * 12000 * 0.5
param_DWN_6_3000 = (2**6) * 3000 + (2**6) * 3000 * 0.5

param_ConvBNN_5 = 1_319_453
param_ConvBNN_64 = 17_338_307
param_BNN_100 = 89810

PARAM_MAP = {
	"DWN_2_4000": int(param_DWN_2_4000),
	"DWN_4_4000": int(param_DWN_4_4000),
	"DWN_6_4000": int(param_DWN_6_4000),
	"DWN_2_16000": int(param_DWN_2_16000),
	"DWN_6_1000": int(param_DWN_6_1000),
	"DWN_4_12000": int(param_DWN_4_12000),
	"DWN_6_12000": int(param_DWN_6_12000),
	"DWN_6_3000": int(param_DWN_6_3000),
	"ConvBNN_5": int(param_ConvBNN_5),
	"ConvBNN_64": int(param_ConvBNN_64),
	"BNN_100": int(param_BNN_100),
}

MAX_NB_FLIP = 5000

def parse_untargeted_filename(file_path: Path) -> dict[str, str | int]:
	stem_parts = file_path.stem.split("_")
	# Expected:
	# bit_flip_untargeted_<task>_<lut_rank>_<layer_size>
	# bit_flip_untargeted_(Conv)BNN_<task>_<layer_size>
	if len(stem_parts) < 6 or stem_parts[:3] != ["bit", "flip", "untargeted"]:
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


def summarize_csv(file_path: Path) -> tuple[float, float, float]:
	base_acc_vals: list[float] = []
	nb_flips_vals: list[float] = []

	with file_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			base_acc_vals.append(float(row["base_acc"]))
			if float(row["nb_flips"]) < MAX_NB_FLIP:
				nb_flips_vals.append(float(row["nb_flips"]))

	if not base_acc_vals:
		raise ValueError(f"No rows in file: {file_path.name}")

	avg_base_acc = mean(base_acc_vals)
	avg_nb_flips = mean(nb_flips_vals)
	std_nb_flips = stdev(nb_flips_vals) if len(nb_flips_vals) > 1 else 0.0
	return avg_base_acc, avg_nb_flips, std_nb_flips


def model_sort_key(model_type: str) -> tuple[int, int]:
	if model_type.startswith("DWN"):
		rank = int(model_type.split("=")[1].rstrip(")"))
		return (0, rank)
	if model_type == "BNN":
		return (1, 0)
	if model_type == "ConvBNN":
		return (2, 0)
	return (3, 0)


def print_markdown_table(task: str, rows: list[dict[str, str | int | float]]) -> None:
	print(f"\n## Task: {task}")
	print("| Model Type | # Parameters | Base Accuracy | Avg nb_flips +/- std |")
	print("|---|---:|---:|---:|")
	for row in rows:
		print(
			"| "
			f"{row['model_type']} | "
			f"{row['parameters']:,} | "
			f"{row['base_acc']:.4f} | "
			f"{row['nb_flips_mean']:.4f} +/- {row['nb_flips_std']:.4f} "
			"|"
		)


def save_table_image(task: str, rows: list[dict[str, str | int | float]], report_dir: Path) -> None:
	headers = ["Model Type", "# Parameters", "Base Accuracy", "Avg nb_flips +/- std"]
	table_data = [
		[
			str(row["model_type"]),
			f"{int(row['parameters']):,}",
			f"{float(row['base_acc'] * 100):.1f}",
			f"{float(row['nb_flips_mean']):.1f} +/- {float(row['nb_flips_std']):.1f}",
		]
		for row in rows
	]

	fig_height = max(2.0, 0.6 + 0.5 * len(table_data))
	fig, ax = plt.subplots(figsize=(12, fig_height))
	ax.axis("off")
	ax.set_title(f"Untargeted Bit Flip Summary - {task}", fontsize=13, fontweight="bold", pad=10)

	table = ax.table(
		cellText=table_data,
		colLabels=headers,
		cellLoc="center",
		colLoc="center",
		loc="center",
	)
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.scale(1, 1.3)

	for (row_index, col_index), cell in table.get_celld().items():
		if row_index == 0:
			cell.set_text_props(weight="bold")
			cell.set_facecolor("#E6EEF8")
		elif row_index % 2 == 0:
			cell.set_facecolor("#F7F9FC")
		else:
			cell.set_facecolor("#FFFFFF")
		cell.set_edgecolor("#C3CBD7")

	output_file = report_dir / f"untargeted_{task}_table.png"
	fig.tight_layout()
	fig.savefig(output_file, dpi=220, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	results_dir = Path(__file__).resolve().parent / "results"
	report_dir = Path(__file__).resolve().parent / "report"
	report_dir.mkdir(parents=True, exist_ok=True)
	files = sorted(results_dir.glob("bit_flip_untargeted_*.csv"))

	if not files:
		print(f"No untargeted result files found in: {results_dir}")
		return

	grouped: dict[str, list[dict[str, str | int | float]]] = {}
	missing_param_keys: set[str] = set()

	for csv_file in files:
		parsed = parse_untargeted_filename(csv_file)
		base_acc, nb_flips_mean, nb_flips_std = summarize_csv(csv_file)
		param_key = str(parsed["param_key"])
		parameters = PARAM_MAP.get(param_key)

		if parameters is None:
			missing_param_keys.add(param_key)
			continue

		row = {
			"model_type": str(parsed["model_type"]),
			"parameters": int(parameters),
			"base_acc": base_acc,
			"nb_flips_mean": nb_flips_mean,
			"nb_flips_std": nb_flips_std,
		}
		grouped.setdefault(str(parsed["task"]), []).append(row)

	for task in sorted(grouped):
		rows = sorted(grouped[task], key=lambda x: model_sort_key(str(x["model_type"])))
		#print_markdown_table(task, rows)
		save_table_image(task, rows, report_dir)
		print(f"Saved image: {report_dir / f'untargeted_{task}_table.png'}")

	if missing_param_keys:
		print("\nMissing parameter mappings for:")
		for key in sorted(missing_param_keys):
			print(f"- {key}")


if __name__ == "__main__":
	main()

