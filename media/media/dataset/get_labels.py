

def add_classes(df, dx_column='dx', classes=None):
    # Given list of 'dx' column values
    dx_lists = df[dx_column].to_list()

    # Flatten the list of lists and remove duplicates
    unique_classes = set([dx for sublist in dx_lists for dx in sublist])
    try: unique_classes.remove('')
    except: pass

    # Se existem classes que nao foram detectadas nas colunas, inicialaiza-las com zero
    if classes is not None:
        for cls in classes:
            if cls not in unique_classes:
                unique_classes.add(cls)
                df[cls] = 0

    # For each unique class, add a new column to the dataframe indicating its presence (1) or absence (0)
    for cls in unique_classes: df[cls] = df[dx_column].apply(lambda x: int(cls in x))
    return df, unique_classes
