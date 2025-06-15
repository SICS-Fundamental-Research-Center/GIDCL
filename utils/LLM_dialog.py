import numpy as np
import pandas as pd
import os
import json # Potentially useful for structured prompts or outputs
from typing import List, Dict, Union, Any
import re
import random

# Assume these functions are defined in utils.func
# from .func import execute_first_function, calculate_f1_with_smoothing, extract_first_function

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

class DetectorRule:
    """
    一个类，用于利用大型语言模型 (LLM) 为数据表中的特定列生成并验证数据清洗规则。

    它通过与 LLM 的多轮对话，根据给定的脏/干净示例对，生成包含正则表达式的 Python 函数，
    并根据 F1 分数进行迭代优化，最终保存有效的函数列表。
    """

    def __init__(self, client, func_model_path: str, detector_thres: float = 0.9):
        """
        初始化 DetectorRule。

        Args:
            client: 用于与 LLM 交互的客户端对象 (e.g., openai.OpenAI 实例)。
            func_model_path (str): LLM 模型的名称或路径。
            detector_thres (float): F1 分数的阈值，当生成的函数达到此F1分数时，认为其有效。
                                    默认为 0.9。
        """
        self.client = client
        self.func_model_path = func_model_path
        self.detector_thres = detector_thres

        # Validate the client object if possible, though exact type depends on actual client.
        if not hasattr(self.client, 'chat') or not hasattr(self.client.chat, 'completions') or \
           not hasattr(self.client.chat.completions, 'create'):
            raise ValueError("提供的 client 对象似乎不是一个有效的 LLM API 客户端 (e.g., OpenAI)。")
        if not isinstance(func_model_path, str) or not func_model_path:
            raise ValueError("func_model_path 必须是一个非空的字符串。")
        if not isinstance(detector_thres, (int, float)) or not (0 <= detector_thres <= 1):
            raise ValueError("detector_thres 必须是介于 0 和 1 之间的浮点数。")

    def generate_and_validate_rules(
        self,
        outlier: np.ndarray,
        outlier_index_list,
        dirty_table: pd.DataFrame,
        clean_table: pd.DataFrame,
        output_path: str,
        dataset_name: str,
        overwrite_func_list: bool = True,
        max_detector_retries: int = 30,
        multi_turn_dialogue: bool = True,
    ) -> List[Dict]:
        """
        根据提供的异常点信息，为脏数据表中的相关列生成并验证 LLM 检测规则。

        Args:
            outlier (np.ndarray): 一个布尔数组，表示每个数据点是否是异常点。
                                  形状应为 (num_samples, num_attributes)。
                                  `outlier.sum(axis=0)` 可以找出哪些列包含异常值。
            dirty_table (pd.DataFrame): 原始的脏数据表。
            clean_table (pd.DataFrame): 对应的干净数据表。
            output_path (str): 用于存储 function_list.npy 的路径。
            dataset_name (str): 数据集名称，用于 LLM 提示。
            overwrite_func_list (bool): 如果为 True，则创建一个新的 function_list。
                                        如果为 False，则加载已有的 function_list.npy
                                        并在其中修改对应异常列的规则。默认为 True。
            max_detector_retries (int): LLM 尝试生成和验证规则的最大次数。默认为 30。

        Returns:
            List[Dict]: 包含生成的检测函数及其元数据的列表。
                        每个字典对应一个列，包含 'detector_prompt_input',
                        'detector_prompt_output', 'detector_prompt_f1'。
                        如果该列没有异常或未成功生成规则，则可能为空或缺失这些键。
        """
        # if not isinstance(outlier, np.ndarray) or outlier.ndim != 2 or outlier.dtype != bool:
        #     raise ValueError("outlier 必须是一个二维布尔型的 NumPy 数组。")
        # if not isinstance(dirty_table, pd.DataFrame) or not isinstance(clean_table, pd.DataFrame):
        #     raise ValueError("dirty_table 和 clean_table 必须是 Pandas DataFrame。")
        # if not isinstance(output_path, str) or not output_path:
        #     raise ValueError("output_path 必须是一个非空的字符串。")
        # if not isinstance(dataset_name, str) or not dataset_name:
        #     raise ValueError("dataset_name 必须是一个非空的字符串。")
        # if not isinstance(overwrite_func_list, bool):
        #     raise TypeError("overwrite_func_list 必须是布尔值。")
        # if not isinstance(max_detector_retries, int) or max_detector_retries <= 0:
        #     raise ValueError("max_detector_retries 必须是正整数。")

        func_list_filepath = os.path.join(output_path, 'function_list.npy')

        # Load existing function_list if not overwriting and file exists
        if not overwrite_func_list and os.path.exists(func_list_filepath):
            print(f"从 {func_list_filepath} 加载现有 function_list。")
            # np.load loads a numpy array, which might contain python objects (dict in this case)
            # allow_pickle=True is necessary for arrays containing Python objects
            function_list = np.load(func_list_filepath, allow_pickle=True).item()
            # Ensure it's a list for consistent manipulation
            # function_list = function_list_array.tolist() if isinstance(function_list_array, np.ndarray) else list(function_list_array)
            # # Ensure function_list has enough capacity for all columns if it's new/small
            # if len(function_list) < dirty_table.shape[1]:
            #     function_list.extend([{} for _ in range(dirty_table.shape[1] - len(function_list))])
        else:
            print("正在创建新的 function_list。")
            # function_list = [{} for _ in range(dirty_table.shape[1])] # Initialize with empty dicts for each column
            function_list = {}
        # Get indices of rows (outliers) that need attention
        # np.where(outlier.sum(axis=1))[0] gives row indices that have at least one outlier
        # outlier_index_list = np.where(outlier.sum(axis=1))[0]


        # Iterate over columns that contain outliers
        # i is the column index
        for i in np.where(np.array(outlier).sum(axis=0))[0]:
            print(f"\n--- 处理列: {dirty_table.columns[i]} (索引: {i}) ---")

            # Initialize or reset the entry for this column in function_list
            # if overwrite_func_list is False, we only re-assign if it's an outlier column
            function_list[i] = {} # This line explicitly resets the dict for the current column if it's an outlier column.

            dirty_list = [] # Stores [clean_value, dirty_value] pairs for mismatched cells
            col_name = clean_table.columns[i]

            # All unique values for this attribute in the dirty table
            clean_list_all_values = list(dirty_table.iloc[:, i].unique())

            # Identify specific dirty/clean pairs from outlier rows for this column
            # Filter rows where clean_table and dirty_table values differ for the current column `i`
            # and where the row itself is an outlier (from outlier_index_list)
            mismatched_rows_for_col = dirty_table.loc[outlier_index_list][dirty_table.loc[outlier_index_list, col_name] != clean_table.loc[outlier_index_list, col_name]]

            for index, row in mismatched_rows_for_col.iterrows():
                dirty_list.append([clean_table.iloc[index, i], dirty_table.iloc[index, i]])

            if dirty_list: # Only proceed if there are actual dirty examples for this column
                # Construct the initial prompt for the LLM
                # detector_inference = (
                #     "The input \n\n%s\n\nare [clean,dirty] cell pairs from table %s column %s, and\n\n%s\n\n "
                #     "are examples of all cells from this columns. Please conclude a general pattern for dirty and clean cells, "
                #     "and write a general function with simple and precise regular expression to detect whether a given cell is dirty or not. "
                #     "Wrap the function within ```python and ``` without test case."
                # ) % (dirty_list, dataset_name, col_name, clean_list_all_values[:20])
                
                detector_inference = "The input \n\n%s\n\nare [clean,dirty] cell pairs from table %s column %s, and\n\n%s\n\n are examples of all cells from this columns. Please conclude a general pattern for dirty and clean cells, and write a general function with simple and precise regular expression to detect whether a given cell is dirty or not, with module re. The function should avoid memorizing specific dirty cases, instead relying on a robust regex pattern to identify general characteristics of dirty values. Wrap the function within ```python and ``` without test case." % (dirty_list,dataset_name,col_name,clean_list_all_values[:20])

                case_prompt = '\n\n Take these cases as examples:\n\n'
                case = ''
                for d in dirty_list:
                    case += "is_dirty('%s')=False\n\nis_dirty('%s')=True\n\n" % (d[0], d[1])

                detector_inference_final = detector_inference + case_prompt + case

                buffer_text = ''
                multi_turn_dialog = ''

                for detector_time in range(max_detector_retries):
                    print(f"尝试生成并验证规则 (列: {col_name}, 尝试次数: {detector_time + 1}/{max_detector_retries})...")
                    
                    try:
                        if multi_turn_dialogue and detector_time < 15:
                            chat_input = detector_inference_final + multi_turn_dialog
                        else:
                            chat_input = detector_inference_final

                        completion = self.client.chat.completions.create(
                            model=self.func_model_path,
                            messages=[           
                                {"role": "user", "content": chat_input},
                            ],
                            timeout = 15
                        )
                        detector_func_raw = completion.choices[0].message.content
                        # print(detector_inference_final + multi_turn_dialog)
                        detector_func_code = detector_func_raw # Assuming extract_first_function is in utils.func
                        
                        if not detector_func_code:
                            raise ValueError("LLM did not return a valid Python function.")

                    except Exception as e:
                        print(f"LLM 交互或函数提取失败: {e}. 重试...")
                        buffer_text = f"\n\n Previous attempt failed to generate a valid function or due to: {e}. Please try again."
                        continue # Continue to next retry

                    TP = 0
                    FP = 0
                    FN = 0
                    TN = 0
                    FP_buffer = [] # Clean detected as True by is_dirty()
                    FN_buffer = [] # Dirty detected as False by is_dirty()

                    # Evaluate the generated function against the known dirty_list
                    for [clean_value, dirty_value] in dirty_list:
                        # Evaluate clean_value
                        try:
                            detector_clean = execute_first_function(detector_func_code, clean_value) # Assuming execute_first_function is in utils.func
                        except Exception as e:
                            detector_clean = True # If function errors on clean value, treat as false positive
                            print(f"Error executing function on clean_value '{clean_value}': {e}")

                        # Evaluate dirty_value
                        try:
                            detector_dirty = execute_first_function(detector_func_code, dirty_value) # Assuming execute_first_function is in utils.func
                        except Exception as e:
                            detector_dirty = False # If function errors on dirty value, treat as false negative
                            print(f"Error executing function on dirty_value '{dirty_value}': {e}")

                        # Calculate TP, FP, FN, TN
                        if not detector_clean: # is_dirty(clean_value) should be False
                            TP += 1 # Correctly classified clean as not dirty (True Negative in binary classification for "is_dirty")
                        else:
                            FP += 1 # Incorrectly classified clean as dirty (False Positive)
                            FP_buffer.append(clean_value)

                        if detector_dirty: # is_dirty(dirty_value) should be True
                            TN += 1 # Correctly classified dirty as dirty (True Positive)
                        else:
                            FN += 1 # Incorrectly classified dirty as not dirty (False Negative)
                            FN_buffer.append(dirty_value)

                    detector_f1 = calculate_f1_with_smoothing(tp=TP, fp=FP, fn=FN) # Assuming calculate_f1_with_smoothing is in utils.func

                    if detector_f1 > self.detector_thres:
                        function_list[i]['detector_prompt_input'] = detector_inference_final + multi_turn_dialog
                        function_list[i]['detector_prompt_output'] = detector_func_raw # Store the raw LLM output
                        function_list[i]['detector_func_code'] = detector_func_code # Store extracted code
                        function_list[i]['detector_prompt_f1'] = detector_f1
                        print(f'函数成功! 数据集: {dataset_name}, 属性: {col_name}, F1: {detector_f1:.4f}, 尝试次数: {detector_time + 1}.')
                        break # Rule successfully generated and validated
                    else:
                        # Add FP/FN Cases to buffer_text for next LLM turn
                        buffer_text = ''
                        for FP_case in FP_buffer:
                            buffer_text += "is_dirty('{}')=False\n\n".format(FP_case)
                        for FN_case in FN_buffer:
                            buffer_text += "is_dirty('{}')=True\n\n".format(FN_case)
                        multi_turn_dialog = "\n\n Previous generation result is {}\n\nHowever it fails to generate the following case\n\n{}.".format(extract_first_function(detector_func_raw),buffer_text)

                        print(f'函数失败! 数据集: {dataset_name}, 属性: {col_name}, F1: {detector_f1:.4f}, 尝试次数: {detector_time + 1}.')
                        if detector_time == max_detector_retries - 1:
                            print(f"达到最大尝试次数 ({max_detector_retries})，未能为列 {col_name} 生成满意函数。")
                        continue # Continue to next retry

            else:
                print(f"列 {col_name} 没有发现需要清洗的脏/净对，跳过生成函数。")
                function_list[i] = {
                    'detector_prompt_input': "No dirty/clean pairs found.",
                    'detector_prompt_output': None,
                    'detector_func_code': None,
                    'detector_prompt_f1': 1.0 # If no dirty cells, assume perfect F1
                }


        # Save the final function_list to .npy file
        np.save(func_list_filepath, function_list)
        print(f"function_list 已保存到: {func_list_filepath}")

        return function_list

    def generate_and_validate_corruption_rules(
        self,
        outlier: np.ndarray,
        outlier_index_list: List[int],
        dirty_table: pd.DataFrame,
        clean_table: pd.DataFrame,
        function_list: Dict[int, Dict[str, Any]],
        dataset_name: str,
        max_generator_retries: int = 30,
        generate_thres: float = 0.8,
        multi_turn_dialogue: bool = False,
    ) -> Dict[int, Dict[str, Any]]:

        for i in np.where(np.array(outlier).sum(axis=0))[0]:
            if(function_list[i]!={}): ## Require the success generation of detector Function!!
                dirty_list = []
                beer_clean_set = clean_table.iloc[outlier_index_list]
                beer_dirty_set = dirty_table.iloc[outlier_index_list]
                col_name = clean_table.columns[i]
                clean_list = list(dirty_table.iloc[:,i].unique()) ## all values for i-th attribute in dirty table
                label_list = list(beer_clean_set.iloc[:,i].unique()) ## all clean values for i-th attribute
                for index,row in beer_dirty_set[beer_dirty_set.iloc[:,i]!=beer_clean_set.iloc[:,i]].iterrows():
                    dirty_list.append([clean_table.iloc[index,i],dirty_table.iloc[index,i]])
                generator_inference_generate = "The input \n\n%s\n\nare [clean,dirty] cell pairs from table %s column %s, and\n\n%s\n\n are regular expressions to detect whether a given cell is dirty or not. Please write a general function Generate() with simple and precise regular expression to corrupt a clean cell to dirty one with random and re. Only return one single function Generate, any other inner function is_dirty should be wrapped inside function Generate. Wrap the function Generate() within ```python and ``` without test case." % (dirty_list,dataset_name,col_name,extract_first_function(function_list[i]['detector_prompt_output']))
                case_prompt = '\n\n Take these cases as examples:\n\n'
                case = ''
                for d in dirty_list:
                    case += "Generate('%s')='%s'\n\n" % (d[0],d[1])
                generator_inference_final = generator_inference_generate + case_prompt + case
                buffer_text = ''
                multi_turn_dialog = ''
                for generate_times in range(30):
                    if multi_turn_dialogue and generate_times <15:
                        chat = generator_inference_final + multi_turn_dialog
                    else:
                        chat = generator_inference_final
                    completion = self.client.chat.completions.create(
                    model=self.func_model_path,
                    messages=[
                        {"role": "user", "content": chat}
                    ],
                    timeout = 15
                    )
                    generator_func = completion.choices[0].message.content
                    count = 0
                    count_base = 0
                    buffer_list = [] ## each element in list is [clean,corrupt_generation]
                    for d in clean_list+label_list: # all value in dirty table
                        if(execute_first_function(function_list[i]['detector_prompt_output'],d)==False): ## The detect function judge the input to be clean, next generate dirty value
                            count_base += 1
                            try:
                                return_dirty = execute_first_function(completion.choices[0].message.content,d)
                                return_detect = execute_first_function(function_list[i]['detector_prompt_output'],return_dirty)
                            except:
                                return_dirty = None
                                return_detect = False
                            
                            if(return_detect==True): ## the generated value is dirty in its regex format, and can be detected
                                count += 1
                            else : ## Generation() successfully generate value
                                buffer_list.append([d,return_dirty])
                                
                    if(count_base==0):
                        generate_f1 = 0
                    else:
                        generate_f1 = count / count_base
                    if generate_f1> generate_thres:
                        print('function success for dataset {} attribute {} with f1:{} at {} times try'.format(dataset_name,col_name,generate_f1,generate_times))
                        function_list[i]['generator_prompt_input'] = generator_inference_final + multi_turn_dialog
                        function_list[i]['generator_prompt_output'] = generator_func
                        function_list[i]['generator_prompt_f1'] = generate_f1
                        break
                    else:
                        buffer_text = ''
                        for [clean_value,corruption_value] in buffer_list:
                            if corruption_value!=None:
                                buffer_text += "Generation('{}')='{}'\n\n".format(clean_value,corruption_value)
                            else:
                                buffer_text += "Generation('{}')\n\n".format(clean_value)
                        # multi_turn_dialog = "\n\n Previous generation result is {}\n\nHowever it fails to corrupt a clean cell to dirty one, with the following base cases:\n\n{}.".format(extract_first_function(generator_func),buffer_text)
                        # print(multi_turn_dialog)
                        multi_turn_dialog = ''
                        print('function fail for dataset {} attribute {} with f1:{} at {} times try'.format(dataset_name,col_name,generate_f1,generate_times))
                    # generate_times += 1
                # print(count / count_base)

        return function_list