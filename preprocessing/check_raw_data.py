import pandas as pd 

df = pd.read_csv("./data/raw/raw_data.csv")
# print(df)

# check dupplicate 
duplicate_plates = (
    df.groupby("plate")
      .size()
      .reset_index(name="count")
      .query("count >= 2")
)
# print(duplicate_plates)

df_remove_dupplicated = df.drop_duplicates(subset="plate", keep="first")
# print(df_remove_dupplicated)

# remove plate nan
df_remove_dupplicated = df_remove_dupplicated.dropna(subset=["plate"])
# print(df_remove_dupplicated)

# check dupplicate after remove
def remove_special_characters(plate: str) -> str:
    return ''.join(e for e in plate if e.isalnum())

df_remove_dupplicated["clean_plate"] = df_remove_dupplicated["plate"].apply(remove_special_characters)
# print(df_remove_dupplicated)

# duplicate_plates = (
#     df_remove_dupplicated.groupby("clean_plate")
#       .size()
#       .reset_index(name="count")
#       .query("count >= 2")
# )
# print(duplicate_plates)

# df_duplicates = (
#     df_remove_dupplicated[df_remove_dupplicated["clean_plate"].duplicated(keep=False)]
#     .sort_values("plate")
# )

# print(df_duplicates)


# rename + create new data 
import os
os.makedirs("./data/processed", exist_ok=True)
# df_remove_dupplicated.to_csv("./data/processed/processed_data2.csv", index=False)

import cv2

# save all images in new folder, remain old name
RAW_IMG_FOLDER = "./data/raw/images"
PROCESSED_IMG_FOLDER = "./data/processed/images"
os.makedirs(PROCESSED_IMG_FOLDER, exist_ok=True)
for idx, row in df_remove_dupplicated.iterrows():
    img_name = row["img"]
    img_path = os.path.join(RAW_IMG_FOLDER, img_name)
    new_img_path = os.path.join(PROCESSED_IMG_FOLDER, img_name)

    # copy image to new folder
    img = cv2.imread(img_path)
    cv2.imwrite(new_img_path, img)