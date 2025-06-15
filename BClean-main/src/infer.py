from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from BClean.src.structure import BN_Structure
import pandas as pd
import re
import time

from concurrent import futures
from tqdm import tqdm

class Inference:
	def __init__(self,
	             dirty_data,
	             data,
	             model,
	             model_dict,
	             attrs,
	             Frequence_list,
	             Occurrence_1,
	             CompensativeParameter,
	             infer_strategy = "PI",
	             chunksize = 1,
	             num_worker = 1,
	             tuple_prun = 1.0):
		self.dirty_data = dirty_data
		self.data = data
		self.model = model
		self.model_dict = model_dict
		self.attrs = attrs
		self.Frequence_list = Frequence_list
		self.Occurrence_1 = Occurrence_1
		self.CompensativeParameter = CompensativeParameter
		self.infer_strategy = infer_strategy
		self.repair_err = {}
		self.chunksize = chunksize
		self.num_worker = num_worker
		self.tuple_prun = tuple_prun

	def Repair(self, attr_type):
		print("++++++++++++++++++++++++error repairring+++++++++++++++++++++++++++++")
		nodes_list = []
		for at in attr_type:
			nodes_list.append(at)
		data = self.data
		model = self.model
		model_dict = self.model_dict
		data = data[list(nodes_list)].copy()
		repair_data = data.copy()
		# clean_data = clean_data[list(nodes_list)].copy()
		
		nodes_list_sort = list(sorted(nodes_list, key = lambda x: len(model.get_parents(x))))
		self.CompensativeParameter.init_tf_idf(nodes_list_sort)
		lines = [i for i in range(repair_data.shape[0])]
		print(len(lines))
		args = [[repair_data.iloc[[line]].copy(
		), line, model, model_dict, nodes_list_sort, attr_type] for line in lines]
		res = []
		chunksize = self.chunksize
		result = pd.DataFrame(columns = data.columns)
		num_worker = self.num_worker
		with futures.ProcessPoolExecutor(max_workers = num_worker) as executor:
			for line, r in zip(lines, executor.map(self.repair_line, args, chunksize = chunksize)):
				res.append(r)
				result = result.append(r["data"], ignore_index = True)
				self.repair_err = {**self.repair_err, **r["repairdict"]}
				if ((line + 1) % 100 == 0):
					print('{} tuple repairing done'.format(line + 1))
			
		return {"dirty":self.dirty_data, "result": result, "repair_err": self.repair_err}
	
	def repair_line(self, args):
		def compute(v, model, node, line):
			result1 = 1
			cpd = model.get_cpds(node)
			values = cpd.values
			state_names = cpd.state_names
			P = values
			for attr in state_names:
				if (attr == node):
					ind = state_names[attr].index(v)
					P = P[ind]
				else:
					ind = state_names[attr].index(data_line.loc[line, attr])
					P = P[ind]
			result1 *= P
			
			result2 = 1
			for child in children:
				if (data_line.loc[line, child] == "A Null Cell"):
					continue
				
				cpd_child = model.get_cpds(child)
				values_child = cpd_child.values
				state_names_child = cpd_child.state_names
				P_child = values_child
				for attr in state_names_child:
					if (attr == node):
						ind = state_names_child[attr].index(v)
						P_child = P_child[ind]
					else:
						ind = state_names_child[attr].index(
							data_line.loc[line, attr])
						P_child = P_child[ind]
				result2 *= P_child
			t = result1 * result2
			
			return t
		
		repair_err = {}
		data_line, line, model_all, model_list, nodes_list_sort, attr_type = args
		nodes_list_sort = [attr for attr in self.prun(data_line, line, attr_type, nodes_list_sort)]
		for val in nodes_list_sort:
			obs = data_line.loc[line, val]
			model = model_list[val]
			if (val in attr_type):
				if (attr_type[val]["repairable"] == "N"):
					continue
			ty = attr_type[val]
			if (self.infer_strategy == "org"):
				predict_data_drop = data_line.copy()
				predict_data_drop.drop(columns = val, axis = 1, inplace = True)
				pred = model_all.predict_probability(predict_data_drop)
				dom = self.data[val].unique().tolist()
				p_obs_G_cand = self.CompensativeParameter.return_penalty(obs = obs,
				                                                         attr = val,
				                                                         index = line,
				                                                         data_line = data_line,
				                                                         prior = dom)
				total = [(v[len(val) + 1: ], p_obs_G_cand[v[len(val) + 1: ]] * (pred[v].values)[0]) for k, v in enumerate(pred)]
			else:
				parents = model.get_parents(val)
				children = model.get_children(val)
				
				# ==================== numerical ====================
				if (len(parents) == 0 and len(children) == 0):
					if (self.data.loc[line, val] != self.dirty_data.loc[line, val]):
						repair_err[(line, val)] = self.data.loc[line, val]
					continue
				
				# ==================== domain prune ====================
				if (self.infer_strategy == "PIPD"):
					dom = self.CompensativeParameter.return_penalty_test(obs = obs,
					                                            attr = val,
					                                            index = line,
					                                            data_line = data_line,
					                                            attr_order = nodes_list_sort,
					                                            prior = self.data[val].unique().tolist())
				# ==================== domain prune ====================
				else:
					dom = self.data[val].unique().tolist()
				
				p_obs_G_cand = self.CompensativeParameter.return_penalty(obs = obs,
				                                           attr = val,
				                                           index = line,
				                                           data_line = data_line,
				                                           prior = dom)
				
				pred_cand = {v: compute(v, model, val, line) for v in dom}
				total = [(value, pred_cand[value] * p_obs_G_cand[value]) for value in pred_cand]
			
			total.sort(key = lambda x: x[1], reverse = True)
			for cell in total:
				if ((attr_type[val]["pattern"] == None or (
						attr_type[val]["pattern"] != None and re.search(attr_type[val]["pattern"], cell[0])))
						and (attr_type[val]["AllowNull"] == "Y" or (
								attr_type[val]["AllowNull"] == "N" and str(cell[0]) != "A Null Cell"))):
					data_line.loc[line, val] = str(cell[0])
					if (str(cell[0]) != self.dirty_data.loc[line, val]):
						repair_err[(line, val)] = str(cell[0])
					break
		
		return {"id": line, "data": data_line, "repairdict": repair_err}
	
	def prun(self, data_line, line, attr_type, node_list):
		res = []
		Freq, Cor = self.Frequence_list, self.Occurrence_1
		for attr_obs in node_list:
			val_obs = data_line.loc[line, attr_obs]
			cooccurance_dist = []
			for attr_cor in node_list:
				if (attr_obs == attr_cor):
					continue
				val_other = data_line.loc[line, attr_cor]
				val_cor = Cor[attr_obs][val_obs][attr_cor][val_other]
				val_freq = Freq[attr_cor][val_other]
				score = val_cor / val_freq
				cooccurance_dist.append(score)
			
			P = sum(cooccurance_dist) / len(cooccurance_dist)
			if (P < self.tuple_prun):
				res.append(attr_obs)
		
		return res
	
