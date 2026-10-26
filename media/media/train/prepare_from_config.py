import os
from pathlib import Path

from media.dataset.build_csv import build_csv
from media.dataset.get_labels import add_classes
from media.dataset.validation.stratified import stratified_group_k_fold, add_folds
from media.scripts.resize_images import walk_and_resize
from media.helpers.create_dir import create_dir
import gc

def resized_image_path(image_path, resized_path, original_path):
    """Preserve a record's relative path on both Windows and POSIX systems."""
    relative = Path(image_path).relative_to(Path(original_path))
    return str(Path(resized_path) / relative)

def prepare_resize(config):
    '''Resize images
    '''
    path = create_dir(os.path.join(config.save_dir, 'resized_images'))
    walk_and_resize(config.data_img_dir,
                    path,
                    (config.size, config.size),
                    'resize')
    return path

def prepare_df(config):
    '''Build CSV
    '''
    df = build_csv(config.data_img_dir)


    if config.resized_path is not None:
        df['original_image_full_path'] = df['image_full_path']
        df['image_full_path'] = df['image_full_path'].apply(
            lambda x: resized_image_path(x, config.resized_path, config.data_img_dir))
    df, unique_labels = add_classes(df, dx_column='dx', classes=config.classes)
    df['is_norm'] = df['dx'].apply(lambda x: 'NORM' in x)
    df['Normal'] = df['dx'].apply(lambda x: 'NORM' in x)
    # df['Normal2'] = df['Normal']
    df = df.sample(frac=config.data_perc, random_state=42).reset_index(drop=True)

    '''Validation
    '''
    kfold = stratified_group_k_fold(df, config.classes, config.group_column, n_splits=config.kfold_splits)
    df = add_folds(df, kfold)
    gc.collect()
    return df

def build_test_set(df, save_dir):
    '''Build test set
    '''
    df = df[df['fold_0_valid'] == False]
    df.to_csv(os.path.join(save_dir, 'df_test.csv'), index=False)
    return df
