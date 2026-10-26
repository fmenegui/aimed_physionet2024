""" build_csv.py
Código para gerar CSV que mapeia IMAGEM e LABEL a partir de um diretório contendo arquivos .hea e .png
"""

import os
from tqdm import tqdm
import pandas as pd


def to_float(x):
    try:
        return float(x)
    except:
        return float("nan")


def get_variables(string, variable_name, sep=","):
    variables = []
    has_variable = False
    for line in string.split("\n"):
        if line.startswith(variable_name):
            variables.extend(
                [
                    variable.strip()
                    for variable in line[len(variable_name) :].strip().split(sep)
                ]
            )
            has_variable = True
    return variables, has_variable


def get_header_file(record):
    return record + ".hea" if not record.endswith(".hea") else record

def load_record(record):
    header_name = get_header_file(record)
    header_full_path = os.path.join(data_dir, header_name)
    base_path = os.path.dirname(header_full_path)
    header_content = load_header(header_full_path)

    images_files = get_image_files_from_header(header_content)
    images = []
    for image in images_files: images.append(os.path.join(base_path, image))
    return images

def load_text(filename):
    with open(filename, "r") as file:
        return file.read()


def load_header(record):
    header_file = get_header_file(record)
    return load_text(header_file)


def get_dxs_from_header(header_string):
    dxs, has_dx = get_variables(header_string, "#Dx:")
    if not has_dx: dxs, has_dx = get_variables(header_string, "# Dx:")
    if not has_dx: dxs, has_dx = get_variables(header_string, "# Dx:")
    if not has_dx: dxs, has_dx = get_variables(header_string, "# #Dx:")
    if not has_dx: dxs, has_dx = get_variables(header_string, "# Labels:")
    if not has_dx:
        raise Exception("No dx classes available")
    return dxs


def load_dx(record):
    header = load_header(record)
    return get_dxs_from_header(header)


def load_dxs(record):
    return load_dx(record)


def get_image_files_from_header(header_string):
    images, has_image = get_variables(header_string, "#Image:")
    if not has_image: images, has_image = get_variables(header_string, "# Image:")
    if not has_image: raise Exception("No images available")
    return images


def get_image_files(record):
    header = load_header(record)
    return get_image_files_from_header(header)


def get_sex_from_header(header):
    sex, has_sex = get_variables(header, "#Sex:")
    if not has_sex: sex, has_sex = get_variables(header, "# Sex:")
    if not has_sex: sex, has_sex = get_variables(header, "# #Sex:")
    return (
        sex[0].lower()
        if (has_sex and sex[0].lower() in ("male", "female"))
        else "unknown"
    )


def get_age_from_header(header):
    age, has_age = get_variables(header, "#Age:")
    if not has_age: age, has_age = get_variables(header, "# Age:")
    if not has_age: age, has_age = get_variables(header, "# #Age:")
    return to_float(age[0]) if has_age else float("nan")


def get_height_from_header(header):
    height, has_height = get_variables(header, "#Height:")
    if not has_height: height, has_height = get_variables(header, "# Height:")
    if not has_height: height, has_height = get_variables(header, "# #Height:")
    return to_float(height[0]) if has_height else float("nan")


def get_weight_from_header(header):
    weight, has_weight = get_variables(header, "#Weight:")
    if not has_weight: weight, has_weight = get_variables(header, "# Weight:")
    if not has_weight: weight, has_weight = get_variables(header, "# #Weight:")
    return to_float(weight[0]) if has_weight else float("nan")


def find_records(folder):
    records = set()
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".hea"):
                record = os.path.relpath(os.path.join(root, file), folder)[:-4]
                records.add(record)
    return sorted(records)


def build_csv(data_img_dir):
    data_dir = data_img_dir

    df_list = []
    for record in tqdm(find_records(data_dir)):

        # Header
        header_name = get_header_file(record)
        header_base = record
        header_full_path = os.path.join(data_dir, header_name)
        base_path = os.path.dirname(header_full_path)
        header_content = load_header(header_full_path)

        # Image
        images = get_image_files_from_header(header_content)
        if not images or not all(images):
            raise ValueError(f"No images listed in {header_full_path}")

        # Metadata
        dx = get_dxs_from_header(header_content)
        age = get_age_from_header(header_content)
        sex = get_sex_from_header(header_content)
        height = get_height_from_header(header_content)
        weight = get_weight_from_header(header_content)

        for image in images:
            df_list.append(
                {
                    "header_name": header_name,
                    "header_base": header_base,
                    "image_name": image,
                    "dx": dx,
                    "sex": sex,
                    "age": age,
                    "height": height,
                    "weight": weight,
                    "image_full_path": os.path.join(base_path, image),
                }
            )

    if not df_list:
        raise ValueError(f"No labeled ECG image records found in {data_dir}")
    df = pd.DataFrame(df_list)
    df_sorted = df.sort_values(by='header_name')
    df_shuffled = df_sorted.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df_shuffled
    return df
