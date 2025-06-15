from typing import Dict, List
import random
import re
import pandas as pd
import json
import numpy as np
import types
import yaml
def extract_and_make_callable(text: str, function_name: str = None):
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if not match:
        raise ValueError("No Python code block (```python...```) found in the text.")

    code_block = match.group(1).strip()

    local_namespace = {}
    try:
        exec(code_block, globals(), local_namespace)
    except Exception as e:
        raise ValueError(f"Error executing extracted code block: {e}")

    if function_name:
        if function_name not in local_namespace or not isinstance(local_namespace[function_name], types.FunctionType):
            raise ValueError(f"Function '{function_name}' not found or is not a callable function in the extracted code.")
        return local_namespace[function_name]
    else:
        for name, obj in local_namespace.items():
            if isinstance(obj, types.FunctionType):
                return obj
        raise ValueError("No function definition found within the extracted Python code block.")

def cluster_by_attribute(df: pd.DataFrame, cluster_number: int, skip_index = True) -> Dict[int, Dict[str, List[int]]]:
    if df.empty:
        return {}

    if skip_index:
        searchable_columns = df.columns[1:]
    else:
        searchable_columns = df.columns

    if len(searchable_columns) == 0:
        raise ValueError("DataFrame must contain at least two columns (one index + one attribute).")

    best_column = None
    min_diff = float('inf')

    for col in searchable_columns:
        unique_count = df[col].nunique()
        diff = abs(unique_count - cluster_number)
        if diff < min_diff:
            min_diff = diff
            best_column = col
        elif diff == min_diff and best_column is not None and unique_count > df[best_column].nunique():
            best_column = col
        elif best_column is None:
            min_diff = diff
            best_column = col


    if best_column is None:
        raise RuntimeError("No suitable column found for clustering. Check DataFrame structure and column types.")

    print(f"Selected column '{best_column}' for clustering with {df[best_column].nunique()} unique values.")

    unique_values = df[best_column].unique()
    num_actual_clusters = len(unique_values)

    value_to_cluster_id = {value: i for i, value in enumerate(unique_values)}

    clusters: Dict[int, Dict[str, List[int]]] = {i: {'members': []} for i in range(num_actual_clusters)}

    for index, row_value in df[best_column].items():
        cluster_id = value_to_cluster_id[row_value]
        clusters[cluster_id]['members'].append(index)

    return clusters

def split_into_clusters(elements, number_of_clusters=20):
    
    clusters = {}
    

    for i in range(number_of_clusters):
        clusters[i] = {'members': []}
    
    for element in elements:
        random_cluster = random.randint(0, number_of_clusters - 1)
        clusters[random_cluster]['members'].append(element)
    
    return clusters
def sort_clusters_by_members_length(clusters):
    n = len(clusters)
    for i in range(n):
        max_idx = i
        for j in range(i + 1, n):
            if len(clusters[j]['members']) > len(clusters[max_idx]['members']):
                max_idx = j
        clusters[i], clusters[max_idx] = clusters[max_idx], clusters[i]
        
def extract_first_function(input_str):
    function_pattern = re.compile(r'def .+?```', re.DOTALL)
    match = function_pattern.search(input_str)
    if match:
        func_str = match.group()[:-3].strip()
        return func_str
    else:
        return "No function definition found."

def execute_first_function(input_str, cell):
    function_pattern = re.compile(r'def .+?:.*?return .+?[^`]```', re.DOTALL)
    match = function_pattern.search(input_str)
    if match:
        func_def = match.group()[:-3].strip()
        exec_globals = {'re': re,'random': random}
        exec(func_def, exec_globals)
        func_name_pattern = re.compile(r'def (\w+)\(')
        func_name_match = func_name_pattern.search(func_def)
        if func_name_match:
            func_name = func_name_match.group(1)
            is_dirty = exec_globals[func_name]
            result = is_dirty(cell)
            return result
        else:
            print("Function name could not be extracted.")
            return False
    else:
        print("No function definition found.")
        return False
    
def calculate_f1_with_smoothing(tp, fp, fn, epsilon=1e-7):
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return f1

def find_unique_false_rows(df, col, function_list, n):
    col_name = df.columns[col]
    column_data = df[col_name]
    
    detect = extract_and_make_callable(function_list[col]['detector_prompt_output'])
    false_mask = ~column_data.apply(lambda x: detect(x))
    
    unique_values = column_data[false_mask].drop_duplicates()
    
    if len(unique_values) > n:
        unique_values = unique_values.iloc[:n]
    result_indices = []
    for value in unique_values:
        index = list(df[df[col_name] == value].index)
        result_indices.extend(index)
    if len(result_indices) >= n:
        result_indices = np.random.choice(result_indices, n, replace=False)
    return result_indices

def find_element_in_clusters(clusters, element):
    for key, cluster in clusters.items():
        if element in cluster['members']:
            return cluster['members']
    
    return None

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data