import pandas as pd
import numpy as np
from itertools import permutations

def _weighted_entropy(series: pd.Series, weights: np.ndarray, total_weight: float) -> float:
    value_weights = series.groupby(series).apply(lambda s: weights[s.index].sum())
    probabilities = value_weights / total_weight
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def find_categorical_dependencies(df: pd.DataFrame, coreset_indices: list, alpha: float = 1.0, threshold: float = 0.95) -> dict:
    if not all(df[col].dtype == 'object' for col in df.columns):
        print("Warning: Not all columns are of string/object type. Converting them for analysis.")
        df = df.astype(str)

    weights = np.full(len(df), 1.0 / alpha)
    valid_coreset_indices = [i for i in coreset_indices if i < len(df)]
    weights[valid_coreset_indices] = 1.0
    total_weight = np.sum(weights)
    
    df_weighted = df.copy()
    df_weighted['__weights__'] = weights

    column_pairs = list(permutations(df.columns, 2))
    
    dependencies = {}

    entropies = {col: _weighted_entropy(df[col], weights, total_weight) for col in df.columns}

    for col_x, col_y in column_pairs:
        h_y = entropies[col_y]
        
        if h_y == 0:
            score = 1.0
        else:
            joint_probs = df_weighted.groupby([col_x, col_y])['__weights__'].sum() / total_weight
            joint_probs = joint_probs[joint_probs > 0]
            h_xy = -np.sum(joint_probs * np.log2(joint_probs))

            h_x = entropies[col_x]
            h_y_given_x = h_xy - h_x
            
            score = (h_y - h_y_given_x) / h_y
            
        score = max(0, min(1, score)) 
        if score > threshold:
            dependency_key = f"{col_x} -> {col_y}"
            dependencies[dependency_key] = score
            
    return dependencies

def correct_dataframe(
    correction_df: pd.DataFrame, 
    dependencies: list, 
    training_df: pd.DataFrame, 
    coreset_indices: list, 
    alpha: float = 1.0
) -> pd.DataFrame:
    df_corrected = correction_df.copy()
    
    weights = np.full(len(training_df), 1.0 / alpha)
    valid_coreset_indices = [i for i in coreset_indices if i < len(training_df)]
    weights[valid_coreset_indices] = 1.0
    
    df_weighted_training = training_df.copy()
    df_weighted_training['__weights__'] = weights
    
    master_correction_map = {}
    
    for dep in dependencies:
        try:
            col_A, col_B = dep.split(' -> ')
        except ValueError:
            print(f"Skipping invalid dependency format: {dep}")
            continue
            
        if col_A not in training_df.columns or col_B not in training_df.columns:
            print(f"Skipping dependency '{dep}' as columns are not in training_df.")
            continue
        
        weighted_counts = df_weighted_training.groupby([col_A, col_B])['__weights__'].sum()
        
        vote_winner_df = weighted_counts.reset_index()
        idx = vote_winner_df.groupby(col_A)['__weights__'].idxmax()
        final_map_df = vote_winner_df.loc[idx]

        correction_map = pd.Series(final_map_df[col_B].values, index=final_map_df[col_A]).to_dict()
        master_correction_map[dep] = correction_map

    print("Applying corrections...")
    for dep, correction_map in master_correction_map.items():
        col_A, col_B = dep.split(' -> ')
        
        if col_A not in df_corrected.columns or col_B not in df_corrected.columns:
            print(f"Skipping correction for '{dep}' as columns are not in correction_df.")
            continue
        
        correct_values = df_corrected[col_A].map(correction_map)
        
        is_incorrect = df_corrected[col_B] != correct_values
        can_be_corrected = correct_values.notna()
        rows_to_correct = is_incorrect & can_be_corrected
        
        num_corrections = rows_to_correct.sum()
        if num_corrections > 0:
            print(f"- Applying {num_corrections} corrections for rule '{dep}'")
            df_corrected.loc[rows_to_correct, col_B] = correct_values[rows_to_correct]
        else:
            print(f"- No corrections needed for rule '{dep}'")
            
    return df_corrected