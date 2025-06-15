from typing import Dict, List
import random
import re
import pandas as pd
import json
import numpy as np
import types
import yaml
def extract_and_make_callable(text: str, function_name: str = None):
    """
    Extracts a Python function defined within a markdown code block (```python...```)
    from a given text and makes it callable.

    Args:
        text (str): The input text containing the function definition.
        function_name (str, optional): The name of the function to extract.
                                       If None, the first function definition found
                                       within a Python code block will be extracted.

    Returns:
        callable: The extracted and callable Python function.

    Raises:
        ValueError: If no Python code block is found, or if no function definition
                    is found within the code block, or if a specific function_name
                    is provided but not found.
    """
    # 1. Extract content between ```python and ```
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if not match:
        raise ValueError("No Python code block (```python...```) found in the text.")

    code_block = match.group(1).strip()

    # 2. Compile and execute the code in a temporary namespace
    local_namespace = {}
    try:
        exec(code_block, globals(), local_namespace)
    except Exception as e:
        raise ValueError(f"Error executing extracted code block: {e}")

    # 3. Find and return the function
    if function_name:
        if function_name not in local_namespace or not isinstance(local_namespace[function_name], types.FunctionType):
            raise ValueError(f"Function '{function_name}' not found or is not a callable function in the extracted code.")
        return local_namespace[function_name]
    else:
        # If no specific function name is given, return the first function object found
        for name, obj in local_namespace.items():
            if isinstance(obj, types.FunctionType):
                # print(f"No specific function name provided. Returning the first found function: '{name}'")
                return obj
        raise ValueError("No function definition found within the extracted Python code block.")

def cluster_by_attribute(df: pd.DataFrame, cluster_number: int, skip_index = True) -> Dict[int, Dict[str, List[int]]]:
    """
    Clusters DataFrame rows based on a selected attribute. The attribute chosen
    is the one whose number of unique values is closest to `cluster_number`.
    The 'members' list in the output will contain the integer indices (row numbers)
    of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame. The first column is assumed to be
                           an index column and will be skipped during attribute selection.
        cluster_number (int): The desired number of clusters.

    Returns:
        Dict[int, Dict[str, List[int]]]: A dictionary where keys are cluster IDs (0 to N-1)
                                         and values are dictionaries containing a 'members' list.
                                         Each 'member' is an integer index (row number) from the
                                         original DataFrame.
    """
    if df.empty:
        return {}

    # Skip the first column, assuming it's an index column
    if skip_index:
        searchable_columns = df.columns[1:]
    else:
        searchable_columns = df.columns

    if len(searchable_columns) == 0:
        raise ValueError("DataFrame must contain at least two columns (one index + one attribute).")

    best_column = None
    min_diff = float('inf')

    # Find the column with unique value count closest to cluster_number
    for col in searchable_columns:
        unique_count = df[col].nunique()
        diff = abs(unique_count - cluster_number)
        if diff < min_diff:
            min_diff = diff
            best_column = col
        # If difference is same, prefer column with more unique values (closer to desired)
        elif diff == min_diff and best_column is not None and unique_count > df[best_column].nunique():
            best_column = col
        # Special case for the very first suitable column found
        elif best_column is None:
            min_diff = diff
            best_column = col


    if best_column is None:
        raise RuntimeError("No suitable column found for clustering. Check DataFrame structure and column types.")

    print(f"Selected column '{best_column}' for clustering with {df[best_column].nunique()} unique values.")

    # Get the unique values from the selected column
    unique_values = df[best_column].unique()
    num_actual_clusters = len(unique_values)

    # Map unique values to cluster IDs (0 to num_actual_clusters - 1)
    value_to_cluster_id = {value: i for i, value in enumerate(unique_values)}

    clusters: Dict[int, Dict[str, List[int]]] = {i: {'members': []} for i in range(num_actual_clusters)}

    # Iterate through DataFrame rows by their index and the chosen column's value
    for index, row_value in df[best_column].items():
        cluster_id = value_to_cluster_id[row_value]
        clusters[cluster_id]['members'].append(index) # Append the DataFrame index

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
    """
    对 clusters 进行排序，使得其 'members' 列表的长度按降序排列。
    使用选择排序算法通过交换实现。
    
    :param clusters: 字典，其键为索引，值为另一个字典，该内部字典包含一个键 'members'，其值是一个列表。
    """
    n = len(clusters)
    # 遍历每一个位置，选择剩余部分最大元素的索引
    for i in range(n):
        max_idx = i
        # 找到从 i 到 n-1 中 'members' 最长的索引
        for j in range(i + 1, n):
            if len(clusters[j]['members']) > len(clusters[max_idx]['members']):
                max_idx = j
        # 交换当前 i 和找到的最大值 max_idx
        clusters[i], clusters[max_idx] = clusters[max_idx], clusters[i]
        
def extract_first_function(input_str):
    # Regular expression pattern to extract the first function definition
    # It looks for the 'def' keyword followed by any characters and ending with triple backticks
    function_pattern = re.compile(r'def .+?```', re.DOTALL)
    match = function_pattern.search(input_str)
    if match:
        # Extract the function definition string
        func_str = match.group()[:-3].strip()  # Remove ending backticks and trim whitespace
        return func_str
    else:
        return "No function definition found."

def execute_first_function(input_str, cell):
    # Extract the first occurrence of function definition using regular expression
    function_pattern = re.compile(r'def .+?:.*?return .+?[^`]```', re.DOTALL)
    match = function_pattern.search(input_str)
    if match:
        # Extract the function definition
        func_def = match.group()[:-3].strip()  # Remove ending backticks and trim whitespace
        # Define a context to execute the function within, including necessary modules
        exec_globals = {'re': re,'random': random}
        # Execute the extracted function definition
        exec(func_def, exec_globals)
        # Extract the function name
        func_name_pattern = re.compile(r'def (\w+)\(')
        func_name_match = func_name_pattern.search(func_def)
        if func_name_match:
            func_name = func_name_match.group(1)
            # Execute the function with 'cell' argument
            is_dirty = exec_globals[func_name]
            result = is_dirty(cell)
            # print(f'The function {func_name} was executed with the cell value "{cell}" and returned: {result}')
            return result
        else:
            print("Function name could not be extracted.")
            return False
    else:
        print("No function definition found.")
        return False
    
def calculate_f1_with_smoothing(tp, fp, fn, epsilon=1e-7):
    """
    计算带小修正项的F1分数，避免除以零的情况
    
    参数:
        tp (int/float): 真正例数量
        fp (int/float): 假正例数量
        fn (int/float): 假反例数量
        epsilon (float): 很小的修正项，默认1e-7
    
    返回:
        float: F1分数
    """
    # 计算带修正项的Precision和Recall
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    
    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return f1

def find_unique_false_rows(df, col, function_list, n):
    """
    返回符合给定条件的行序号列表。
    
    :param df: 输入的 DataFrame
    :param col_name: 要检查的列的名称
    :param detect: 一个函数，输入列名和列值，返回布尔值
    :param n: 需要返回的最大行数
    :return: 包含行索引的列表
    """
    # 获取目标列的数据
    col_name = df.columns[col]
    column_data = df[col_name]
    
    # 应用 detect 函数并取反，找到返回 False 的元素
    # 这里我们假设 detect 函数定义为 detect(col_name, value)，返回布尔值
    detect = extract_and_make_callable(function_list[col]['detector_prompt_output'])
    false_mask = ~column_data.apply(lambda x: detect(x))
    
    # 使用 false_mask 过滤数据并去重
    unique_values = column_data[false_mask].drop_duplicates()
    # 
    # 如果 unique_values 的长度大于 n，则截断列表
    if len(unique_values) > n:
        unique_values = unique_values.iloc[:n]
    # 找到这些 unique_values 在原 DataFrame 中的行序号
    result_indices = []
    for value in unique_values:
        # 找到第一个匹配的行索引
        index = list(df[df[col_name] == value].index)
        result_indices.extend(index)
        # if len(result_indices) >= n:
        #     break
    if len(result_indices) >= n:
        result_indices = np.random.choice(result_indices, n, replace=False)
    return result_indices

def find_element_in_clusters(clusters, element):
    """
    在 clusters 的 'members' 中搜索包含给定元素的列表。
    
    :param clusters: 一个字典，其中每个键对应的值也是一个字典，该内部字典包含一个键 'members'，其值是一个列表。
    :param element: 要搜索的元素
    :return: 包含该元素的列表，如果没有找到则返回 None
    """
    # 遍历每个 cluster
    for key, cluster in clusters.items():
        # 检查 'members' 列表中是否包含给定的 element
        if element in cluster['members']:
            return cluster['members']
    
    # 如果没有找到，返回 None
    return None

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data