from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def remove_special_characters(plate: object) -> str:
	if pd.isna(plate):
		return ""
	return re.sub(r"[^0-9A-Za-z]+", "", str(plate)).upper()


def split_dataframe_by_group(
	df: pd.DataFrame,
	group_column: str,
	train_ratio: float,
	val_ratio: float,
	test_ratio: float,
	random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if not pd.isna(train_ratio + val_ratio + test_ratio):
		total = train_ratio + val_ratio + test_ratio
		if abs(total - 1.0) > 1e-6:
			raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

	unique_groups = df[group_column].dropna().nunique()
	if unique_groups < 3:
		raise ValueError("Need at least 3 unique groups to create train/val/test splits")

	# Split train+val vs test
	first_splitter = GroupShuffleSplit(
		n_splits=1, # one time split
		test_size=test_ratio,
		random_state=random_state,
	)
	train_val_idx, test_idx = next(first_splitter.split(df, groups=df[group_column]))
	train_val_df = df.iloc[train_val_idx].copy()
	test_df = df.iloc[test_idx].copy()

	remaining_ratio = train_ratio + val_ratio
	val_share = val_ratio / remaining_ratio

	# split train vs val
	second_splitter = GroupShuffleSplit(
		n_splits=1,
		test_size=val_share,
		random_state=random_state,
	)
	train_idx, val_idx = next(second_splitter.split(train_val_df, groups=train_val_df[group_column]))
	train_df = train_val_df.iloc[train_idx].copy()
	val_df = train_val_df.iloc[val_idx].copy()

	return train_df, val_df, test_df


def sync_data(
	input_csv: str = "./data/synthetic/synthetic_data.csv",
	output_dir: str = "./data/splits",
	train_ratio: float = 0.7,
	val_ratio: float = 0.15,
	test_ratio: float = 0.15,
	random_state: int = 42,
) -> dict[str, pd.DataFrame]:
	input_path = Path(input_csv)
	if not input_path.exists():
		raise FileNotFoundError(f"Could not find input CSV: {input_csv}")

	df = pd.read_csv(input_path)
	if "plate" not in df.columns:
		raise ValueError("Input CSV must contain a 'plate' column")
	if "img" not in df.columns:
		raise ValueError("Input CSV must contain an 'img' column")

	df = df.copy()
	df["clean_plate"] = df["plate"].apply(remove_special_characters)

	# Build group_id from clean_plate + plate_type
	if "plate_type" in df.columns:
		base_id = df["clean_plate"].where(df["clean_plate"] != "", df.get("source_img", df["img"]))
		df["group_id"] = base_id.astype(str) + "_" + df["plate_type"].astype(str)
	else:
		raise ValueError("Input CSV must contain a 'plate_type' column to build group_id")
	
	df = df[df["group_id"].astype(str).str.len() > 0].copy()

	train_df, val_df, test_df = split_dataframe_by_group(
		df=df,
		group_column="group_id",
		train_ratio=train_ratio,
		val_ratio=val_ratio,
		test_ratio=test_ratio,
		random_state=random_state,
	)

	train_df["split"] = "train"
	val_df["split"] = "val"
	test_df["split"] = "test"

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
	combined_df.to_csv(output_path / "all_splits.csv", index=False)
	train_df.to_csv(output_path / "train.csv", index=False)
	val_df.to_csv(output_path / "val.csv", index=False)
	test_df.to_csv(output_path / "test.csv", index=False)

	return {
		"train": train_df,
		"val": val_df,
		"test": test_df,
		"all": combined_df,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Split plate data by grouped vehicle id to avoid data leakage.")
	parser.add_argument("--input-csv", default="./data/synthetic/synthetic_data.csv", help="Input CSV path")
	parser.add_argument("--output-dir", default="./data/splits", help="Output directory for split CSVs")
	parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
	parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
	parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	sync_data(
		input_csv=args.input_csv,
		output_dir=args.output_dir,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		random_state=args.seed,
	)
