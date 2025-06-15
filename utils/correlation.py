import pandas as pd
import numpy as np
from itertools import permutations

def _weighted_entropy(series: pd.Series, weights: np.ndarray, total_weight: float) -> float:
    """Helper function to calculate weighted entropy."""
    # Group by series values and sum their weights
    value_weights = series.groupby(series).apply(lambda s: weights[s.index].sum())
    # Calculate weighted probabilities
    probabilities = value_weights / total_weight
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    # Calculate entropy
    return -np.sum(probabilities * np.log2(probabilities))

def find_categorical_dependencies(df: pd.DataFrame, coreset_indices: list, alpha: float = 1.0, threshold: float = 0.95) -> dict:
    """
    Finds functional dependencies in categorical data using a weighted Theil's U score.

    This function measures asymmetric relationships, e.g., A -> B.

    Args:
        df (pd.DataFrame): The input dataframe with string/categorical data.
        coreset_indices (list): A list of indices for the coreset rows.
        alpha (float, optional): The hyperparameter for weighting non-coreset rows. Defaults to 1.0.
        threshold (float, optional): The score threshold for dependency. Defaults to 0.95.

    Returns:
        dict: A dictionary of dependencies (e.g., "colA -> colB") with a score 
              greater than the threshold.
    """
    if not all(df[col].dtype == 'object' for col in df.columns):
        print("Warning: Not all columns are of string/object type. Converting them for analysis.")
        df = df.astype(str)

    # 1. Initialize weights
    weights = np.full(len(df), 1.0 / alpha)
    # Ensure coreset indices are within bounds
    valid_coreset_indices = [i for i in coreset_indices if i < len(df)]
    weights[valid_coreset_indices] = 1.0
    total_weight = np.sum(weights)
    
    # Attach weights to the dataframe for easy grouping
    df_weighted = df.copy()
    df_weighted['__weights__'] = weights

    # 2. Get all directed pairs of columns (permutations)
    column_pairs = list(permutations(df.columns, 2))
    
    dependencies = {}

    # Pre-calculate entropy for each column to avoid re-computation
    entropies = {col: _weighted_entropy(df[col], weights, total_weight) for col in df.columns}

    for col_x, col_y in column_pairs:
        h_y = entropies[col_y]
        
        # If the entropy of Y is 0, it means Y is a constant. 
        # Any X perfectly "predicts" a constant Y.
        if h_y == 0:
            score = 1.0
        else:
            # Calculate weighted conditional entropy H(Y|X)
            # This is done by calculating the joint entropy H(X,Y) first.
            # H(Y|X) = H(X,Y) - H(X)
            
            # Weighted Joint Entropy H(X,Y)
            joint_probs = df_weighted.groupby([col_x, col_y])['__weights__'].sum() / total_weight
            joint_probs = joint_probs[joint_probs > 0]
            h_xy = -np.sum(joint_probs * np.log2(joint_probs))

            h_x = entropies[col_x]
            h_y_given_x = h_xy - h_x
            
            # 3. Calculate Theil's U score U(Y|X)
            score = (h_y - h_y_given_x) / h_y
            
        # 4. Check against threshold
        # Clamp score between 0 and 1 to handle potential float precision issues
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
    """
    Corrects a dataframe based on learned functional dependencies using a weighted majority vote.

    Args:
        correction_df (pd.DataFrame): The dataframe to be corrected.
        dependencies (list): A list of dependency strings, e.g., ["MeasureName -> Stateavg"].
        training_df (pd.DataFrame): The original dataframe from which dependencies were learned.
        coreset_indices (list): The list of coreset indices from the training_df.
        alpha (float): The hyperparameter for weighting non-coreset rows.

    Returns:
        pd.DataFrame: A new dataframe with corrected values.
    """
    df_corrected = correction_df.copy()
    
    # 1. Prepare weights for the training dataframe
    weights = np.full(len(training_df), 1.0 / alpha)
    valid_coreset_indices = [i for i in coreset_indices if i < len(training_df)]
    weights[valid_coreset_indices] = 1.0
    
    # Create a temporary dataframe with weights for easy calculation
    df_weighted_training = training_df.copy()
    df_weighted_training['__weights__'] = weights
    
    # 2. Build the correction map from the training data
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
        
        # Perform the weighted majority vote
        # Group by both columns and sum the weights
        weighted_counts = df_weighted_training.groupby([col_A, col_B])['__weights__'].sum()
        
        # For each value in col_A, find the col_B value with the maximum summed weight
        # The result of idxmax() is a multi-index, so we need to extract the winning value
        # We reset_index to make it easier to work with
        vote_winner_df = weighted_counts.reset_index()
        idx = vote_winner_df.groupby(col_A)['__weights__'].idxmax()
        final_map_df = vote_winner_df.loc[idx]

        # Create a simple dictionary map {value_A: winning_value_B}
        correction_map = pd.Series(final_map_df[col_B].values, index=final_map_df[col_A]).to_dict()
        master_correction_map[dep] = correction_map

    # 3. Apply the correction map to the input dataframe
    print("Applying corrections...")
    for dep, correction_map in master_correction_map.items():
        col_A, col_B = dep.split(' -> ')
        
        if col_A not in df_corrected.columns or col_B not in df_corrected.columns:
            print(f"Skipping correction for '{dep}' as columns are not in correction_df.")
            continue
        
        # Get the "correct" values by mapping col_A through our correction map
        correct_values = df_corrected[col_A].map(correction_map)
        
        # Identify rows where the current value in col_B is incorrect AND we have a correction available
        is_incorrect = df_corrected[col_B] != correct_values
        can_be_corrected = correct_values.notna()
        rows_to_correct = is_incorrect & can_be_corrected
        
        num_corrections = rows_to_correct.sum()
        if num_corrections > 0:
            print(f"- Applying {num_corrections} corrections for rule '{dep}'")
            # Apply the correction only on the identified rows
            df_corrected.loc[rows_to_correct, col_B] = correct_values[rows_to_correct]
        else:
            print(f"- No corrections needed for rule '{dep}'")
            
    return df_corrected