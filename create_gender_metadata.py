import os

BASE_DIR = r"C:\Vs_code\Mtech_UAITTI\AITTI\dataset\generated_gender"

OUTPUT_FILE = r"C:\Vs_code\Mtech_UAITTI\AITTI\dataset\gender_dataset.txt"

lines = []

lines.append("profession,attribute,class,filename\n")

for folder in os.listdir(BASE_DIR):

    folder_path = os.path.join(BASE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    folder_lower = folder.lower()

    # --------------------------------------------------
    # Gender
    # --------------------------------------------------

    if "male_" in folder_lower:

        attribute = "male"
        class_name = "man"

    elif "female_" in folder_lower:

        attribute = "female"
        class_name = "woman"

    else:
        continue

    # --------------------------------------------------
    # Profession extraction
    # --------------------------------------------------

    profession = (
        folder_lower
        .replace("a_high-quality_realistic_photo_of_a_", "")
        .replace("male_", "")
        .replace("female_", "")
        .replace("_", " ")
    )

    # --------------------------------------------------
    # Image directory
    # --------------------------------------------------

    image_dir = os.path.join(folder_path, "images")

    if not os.path.exists(image_dir):
        continue

    for image_name in os.listdir(image_dir):

        if image_name.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(image_dir, image_name)

            image_path = image_path.replace("\\", "/")

            line = (
                f"{profession},"
                f"{attribute},"
                f"{class_name},"
                f"{image_path}\n"
            )

            lines.append(line)

# ------------------------------------------------------
# Save TXT
# ------------------------------------------------------

with open(OUTPUT_FILE, "w") as f:

    f.writelines(lines)

print(f"\nSaved metadata file to:\n{OUTPUT_FILE}")

print(f"\nTotal entries: {len(lines)-1}")