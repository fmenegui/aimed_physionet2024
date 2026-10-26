"""
Stratified group K-fold splitting with `stratified_group_k_fold`.
"""

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder


def get_new_labels(y):
    """Convert each multilabel vector to a unique string"""
    yy = ["".join(str(l)) for l in y]
    y_new = LabelEncoder().fit_transform(yy)
    return y_new

def add_folds(df, kfold):
    kfold = list(kfold) if not isinstance(kfold, (list)) else kfold

    for i, fold in enumerate(kfold):
        df.loc[:,'fold_'+str(i)+'_train'] = 0
        df.loc[:,'fold_'+str(i)+'_valid'] = 0
        df.loc[:,'test'] = 0
        df.loc[fold['train'], 'fold_'+str(i)+'_train'] = 1
        df.loc[fold['valid'], 'fold_'+str(i)+'_valid'] = 1
        if 'test' in fold.keys() and (fold['test'] is not None): df.loc[fold['test'],         'test'          ] = 1
    return df


def stratified_group_k_fold(dataframe, label_columns, group_column, n_splits=5):
    """
    Generate indices to split data into training and validation set using
    Stratified Group K-Fold cross-validator

    Parameters:
    dataframe (pd.DataFrame): The dataset containing labels and group information.
    label_columns (list): Column names that are used as labels for stratification.
    group_column (str): Column name that represents the group (e.g., patient_id).
    k (int): Number of folds.

    Yields:
    train_idx (array): Indices of the training set.
    valid_idx (array): Indices of the validation set.
    """
    dataframe = dataframe.reset_index(drop=True)

    # Ensure labels are in a single column as a tuple if there are multiple label columns
    if len(label_columns) > 1:
        labels = list(zip(*[dataframe[col] for col in label_columns]))
    else:
        labels = dataframe[label_columns[0]]
    labels = get_new_labels(labels)

    if group_column is None:
        sgkf = StratifiedKFold(n_splits=n_splits)
        groups = None
    else:
        groups = dataframe[group_column]
        sgkf = StratifiedGroupKFold(n_splits=n_splits)

    for train_idx, valid_idx in sgkf.split(dataframe, labels, groups=groups):
        yield {"train": train_idx, "valid": valid_idx, "test": None}



def stratified_group_k_fold_with_test(
    dataframe, label_columns, group_column, n_splits=5
):
    """
    Generate indices to split data into train, validation, and test sets for each fold using
    Stratified Group K-Fold cross-validator.

    Parameters:
    dataframe (pd.DataFrame): The dataset containing labels and group information.
    label_columns (list): Column names that are used as labels for stratification.
    group_column (str): Column name that represents the group (e.g., patient_id).
    n_splits (int): Number of folds.

    Yields:
    dict: A dictionary with 'train', 'valid', and 'test' keys for each fold.
    """
    dataframe = dataframe.reset_index(drop=True)

    # Ensure labels are in a single column as a tuple if there are multiple label columns
    if len(label_columns) > 1:
        labels = list(zip(*[dataframe[col] for col in label_columns]))
    else:
        labels = dataframe[label_columns[0]]
    labels = get_new_labels(labels)

    if group_column is None:
        sgkf = StratifiedKFold(n_splits=5)
        groups = None
    else:
        groups = dataframe[group_column]
        sgkf = StratifiedGroupKFold(n_splits=5)

    # Split train/validation and test
    train_valid_idx, test_idx = next(sgkf.split(dataframe, labels, groups=groups))
    train_valid_set = dataframe.iloc[train_valid_idx]
    test_set = dataframe.iloc[test_idx]

    # Labels and groups for train/validation set
    if len(label_columns) > 1:
        train_valid_labels = list(
            zip(*[train_valid_set[col] for col in label_columns])
        )
    else:
        train_valid_labels = train_valid_set[label_columns[0]]
    train_valid_labels = get_new_labels(train_valid_labels)
    train_valid_groups = train_valid_set[group_column]

    # K-Fold on train/validation set
    sgkf_train_valid = StratifiedGroupKFold(n_splits=n_splits)
    for train_idx, valid_idx in sgkf_train_valid.split(
        train_valid_set, train_valid_labels, train_valid_groups
    ):
        yield {
            "train": train_valid_set.iloc[train_idx].index,
            "valid": train_valid_set.iloc[valid_idx].index,
            "test": test_set.index,
        }


def check_label_balance(dataframe, label_columns, train_idx, valid_idx, test_idx):
    """
    Check the balance of labels across train, validation, and test splits.

    Parameters:
    dataframe (pd.DataFrame): The dataset containing labels.
    label_columns (list): Column names that are used as labels.
    train_idx, valid_idx, test_idx: Indices for train, validation, and test splits.
    """

    def calculate_proportions(df, label_columns):
        label_counts = df[label_columns].sum()
        total_counts = len(df)
        proportions = label_counts / total_counts
        return proportions

    train_proportions = calculate_proportions(dataframe.iloc[train_idx], label_columns)
    valid_proportions = calculate_proportions(dataframe.iloc[valid_idx], label_columns)
    test_proportions = calculate_proportions(dataframe.iloc[test_idx], label_columns)

    balance_df = pd.DataFrame(
        {
            "Train": train_proportions,
            "Validation": valid_proportions,
            "Test": test_proportions,
        }
    )

    return balance_df


# Usage
# balance_df = check_label_balance(dataframe, label_columns, train_idx, valid_idx, test_idx)
# print(balance_df)
