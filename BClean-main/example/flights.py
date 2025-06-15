import time
from BClean.BClean import BayesianClean
from BClean.analysis import analysis
from BClean.dataset import Dataset
from BClean.src.UC import UC
import re

if __name__ == '__main__':
	dirty_path = "../dataset/flights/flights_dirty.csv"
	clean_path = "../dataset/flights/flights_clean.csv"
	dataLoader = Dataset()
	dirty_data = dataLoader.get_data(path = dirty_path)
	clean_data = dataLoader.get_data(path = clean_path)
	model_path = None
	model_save_path = None
	fix_edge = [("flight", "sched_dep_time"), ("flight", "act_dep_time"), ("flight", "sched_arr_time"), ("flight", "act_arr_time")]

	attr = UC(dirty_data)

	attr.build_from_json("../UC_json/flights.json")
	
	# general_pattern = re.compile(r'([1-9]:[0-5][0-9][\s][ap].[m].|1[0-2]:[0-5][0-9][\s][ap].[m].|0[1-9]:[0-5][0-9][\s][ap].[m].)')
	
	# attr.build(attr = "src", type = "Categorical", min_v = 2, max_v = 30, null_allow = "N", repairable = "N",
	#            pattern = None)
	# attr.build(attr = "flight", type = "Categorical", min_v = 2, max_v = 30, null_allow = "N", repairable = "N",
	#            pattern = None)
	# attr.build(attr = "sched_dep_time", type = "Categorical", min_v = 8, max_v = 10, null_allow = "N", repairable = "Y",
	#            pattern = general_pattern)
	# attr.build(attr = "act_dep_time", type = "Categorical", min_v = 8, max_v = 10, null_allow = "N", repairable = "Y",
	#            pattern = general_pattern)
	# attr.build(attr = "sched_arr_time", type = "Categorical", min_v = 8, max_v = 10, null_allow = "N", repairable = "Y",
	#            pattern = general_pattern)
	# attr.build(attr = "act_arr_time", type = "Categorical", min_v = 8, max_v = 10, null_allow = "N", repairable = "Y",
	#            pattern = general_pattern)
	
	pat = attr.PatternDiscovery()
	print("pattern discovery:{}".format(pat))
	
	attr = attr.get_uc()
	
	dirty_data = dataLoader.get_real_data(dirty_data, attr_type = attr)
	clean_data = dataLoader.get_real_data(clean_data, attr_type = attr)
	
	start_time = time.perf_counter()
	
	model = BayesianClean(dirty_df = dirty_data,
	                      clean_df = clean_data,
	                      model_path = model_path,
	                      model_save_path = model_save_path,
	                      cpt_est = "Bayes",
	                      attr_type = attr,
	                      fix_edge = fix_edge,
	                      model_choice = "fix",
	                      infer_strategy = "org",
	                      tuple_prun = 1.0,
	                      maxiter = 1,
	                      num_worker = 32,
	                      chunksize = 20,
	                      model_update = False
	                      )
	
	repair_list = model.clean()
	end_time = time.perf_counter()
	dirty_data, repair_data, clean_data = repair_list["dirty"], repair_list["result"], repair_list["clean"]
	actual_error, repair_error = model.actual_error, repair_list["repair_err"]
	P, R, F = analysis(actual_error, repair_error, dirty_data, clean_data)
	
	print("Repair Pre:{:.5f}, Recall:{:.5f}, F1-score:{:.5f}".format(P, R, F))
	print("++++++++++++++++++++time using:{}+++++++++++++++++++++++".format(end_time - start_time))
	print("date:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
	