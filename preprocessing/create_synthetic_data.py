import albumentations as A
import cv2
import pandas as pd
from pathlib import Path


low_bad_transform = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.22, -0.08),
        contrast_limit=(-0.08, 0.08),
        p=0.9,
    ),
    A.MotionBlur(
        blur_limit=(3, 7),
        p=0.9,
    ),
    A.Downscale(
        scale_range=(0.8, 0.95),
        p=0.25,
    ),
    A.ImageCompression(
        quality_range=(65, 90),
        p=0.25,
    ),
])

medium_bad_transform = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, -0.3),
        contrast_limit=(-0.15, 0.15),
        p=0.9,
    ), 
    A.MotionBlur(
        blur_limit=(9, 11),
        p=0.9,
    ),
    A.Downscale(
        scale_range=(0.65, 0.85),
        p=0.45,
    ),
    A.ImageCompression(
        quality_range=(40, 70),
        p=0.55,
    ),
])

high_bad_transform = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.5, -0.3),
        contrast_limit=(-0.25, 0.2),
        p=1.0,
    ),
    A.MotionBlur(
        blur_limit=(11, 21),
        p=0.9,
    ),
    A.Downscale(
        scale_range=(0.3, 0.55),
        p=0.8,
    ),
    A.ImageCompression(
        quality_range=(10, 40),
        p=0.8,
    ),
])

occluded_bad_transform = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.5, -0.2),
        contrast_limit=(-0.25, 0.08),
        p=0.5,
    ),
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.1, 0.3),
        hole_width_range=(0.1, 0.3),
        fill=0,
        p=1.0,
    ),
    A.MotionBlur(
        blur_limit=(3, 13),
        p=0.5,
    ),
    A.Downscale(
        scale_range=(0.55, 0.75),
        p=0.5,
    ),
    A.ImageCompression(
        quality_range=(35, 65),
        p=0.5,
    ),
])


TRANSFORMS = {
    "low": low_bad_transform,
    "medium": medium_bad_transform,
    "high": high_bad_transform,
    "occluded": occluded_bad_transform,
}

def sync_data(
    csv_path: str = "./data/processed/processed_data2.csv",
    raw_image_dir: str = "./data/processed/images_resized",
    output_dir: str = "./data/synthetic",
) -> pd.DataFrame:
    source_df = pd.read_csv(csv_path)
    raw_image_path = Path(raw_image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, record in source_df.iterrows():
        image_name = str(record["img"])
        label = record["plate"]
        input_image_path = raw_image_path / image_name

        image_bgr = cv2.imread(str(input_image_path))
        if image_bgr is None:
            print(f"Skip missing image: {input_image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        stem = Path(image_name).stem

        for variant_name, transform in TRANSFORMS.items():
            augmented_image = transform(image=image_rgb)["image"]
            output_name = f"{stem}_{variant_name}.png"
            output_file = output_path / output_name

            augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), augmented_bgr)

            rows.append(
                {
                    "img": output_name,
                    "plate": label,
                    "source_img": image_name,
                    "augment_type": variant_name,
                    "plate_type": record["plate_type"]
                }
            )

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path / "synthetic_data.csv", index=False)
    return output_df


if __name__ == "__main__":
    sync_data()
