import numpy as np
import pandas as pd
import os
import json
from typing import List, Dict, Union, Any
import re
import random

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

class DetectorRule:
    def __init__(self, client, func_model_path: str, detector_thres: float = 0.9):
        self.client = client
        self.func_model_path = func_model_path
        self.detector_thres = detector_thres

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
        func_list_filepath = os.path.join(output_path, 'function_list.npy')

        if not overwrite_func_list and os.path.exists(func_list_filepath):
            print(f"从 {func_list_filepath} 加载现有 function_list。")
            function_list = np.load(func_list_filepath, allow_pickle=True).item()
        else:
            print("正在创建新的 function_list。")
            function_list = {}

        for i in np.where(np.array(outlier).sum(axis=0))[0]:
            print(f"\n--- 处理列: {dirty_table.columns[i]} (索引: {i}) ---")

            function_list[i] = {}

            dirty_list = []
            col_name = clean_table.columns[i]

            clean_list_all_values = list(dirty_table.iloc[:, i].unique())

            mismatched_rows_for_col = dirty_table.loc[outlier_index_list][dirty_table.loc[outlier_index_list, col_name] != clean_table.loc[outlier_index_list, col_name]]

            for index, row in mismatched_rows_for_col.iterrows():
                dirty_list.append([clean_table.iloc[index, i], dirty_table.iloc[index, i]])

            if dirty_list:
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
                            timeout = 60
                        )
                        detector_func_raw = completion.choices[0].message.content
                        detector_func_code = detector_func_raw
                        
                        if not detector_func_code:
                            raise ValueError("LLM did not return a valid Python function.")

                    except Exception as e:
                        print(f"LLM 交互或函数提取失败: {e}. 重试...")
                        buffer_text = f"\n\n Previous attempt failed to generate a valid function or due to: {e}. Please try again."
                        continue

                    TP = 0
                    FP = 0
                    FN = 0
                    TN = 0
                    FP_buffer = []
                    FN_buffer = []

                    for [clean_value, dirty_value] in dirty_list:
                        try:
                            detector_clean = execute_first_function(detector_func_code, clean_value)
                        except Exception as e:
                            detector_clean = True
                            print(f"Error executing function on clean_value '{clean_value}': {e}")

                        try:
                            detector_dirty = execute_first_function(detector_func_code, dirty_value)
                        except Exception as e:
                            detector_dirty = False
                            print(f"Error executing function on dirty_value '{dirty_value}': {e}")

                        if not detector_clean:
                            TP += 1
                        else:
                            FP += 1
                            FP_buffer.append(clean_value)

                        if detector_dirty:
                            TN += 1
                        else:
                            FN += 1
                            FN_buffer.append(dirty_value)

                    detector_f1 = calculate_f1_with_smoothing(tp=TP, fp=FP, fn=FN)

                    if detector_f1 > self.detector_thres:
                        function_list[i]['detector_prompt_input'] = detector_inference_final + multi_turn_dialog
                        function_list[i]['detector_prompt_output'] = detector_func_raw
                        function_list[i]['detector_func_code'] = detector_func_code
                        function_list[i]['detector_prompt_f1'] = detector_f1
                        print(f'函数成功! 数据集: {dataset_name}, 属性: {col_name}, F1: {detector_f1:.4f}, 尝试次数: {detector_time + 1}.')
                        break
                    else:
                        buffer_text = ''
                        for FP_case in FP_buffer:
                            buffer_text += "is_dirty('{}')=False\n\n".format(FP_case)
                        for FN_case in FN_buffer:
                            buffer_text += "is_dirty('{}')=True\n\n".format(FN_case)
                        multi_turn_dialog = "\n\n Previous generation result is {}\n\nHowever it fails to generate the following case\n\n{}.".format(extract_first_function(detector_func_raw),buffer_text)

                        print(f'函数失败! 数据集: {dataset_name}, 属性: {col_name}, F1: {detector_f1:.4f}, 尝试次数: {detector_time + 1}.')
                        if detector_time == max_detector_retries - 1:
                            print(f"达到最大尝试次数 ({max_detector_retries})，未能为列 {col_name} 生成满意函数。")
                        continue

            else:
                print(f"列 {col_name} 没有发现需要清洗的脏/净对，跳过生成函数。")
                function_list[i] = {
                    'detector_prompt_input': "No dirty/clean pairs found.",
                    'detector_prompt_output': None,
                    'detector_func_code': None,
                    'detector_prompt_f1': 1.0
                }

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
            if(function_list[i]!={}):
                dirty_list = []
                beer_clean_set = clean_table.iloc[outlier_index_list]
                beer_dirty_set = dirty_table.iloc[outlier_index_list]
                col_name = clean_table.columns[i]
                clean_list = list(dirty_table.iloc[:,i].unique())
                label_list = list(beer_clean_set.iloc[:,i].unique())
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
                    timeout = 60
                    )
                    generator_func = completion.choices[0].message.content
                    count = 0
                    count_base = 0
                    buffer_list = []
                    for d in clean_list+label_list:
                        if(execute_first_function(function_list[i]['detector_prompt_output'],d)==False):
                            count_base += 1
                            try:
                                return_dirty = execute_first_function(completion.choices[0].message.content,d)
                                return_detect = execute_first_function(function_list[i]['detector_prompt_output'],return_dirty)
                            except:
                                return_dirty = None
                                return_detect = False
                            
                            if(return_detect==True):
                                count += 1
                            else :
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
                        multi_turn_dialog = ''
                        print('function fail for dataset {} attribute {} with f1:{} at {} times try'.format(dataset_name,col_name,generate_f1,generate_times))
        return function_list