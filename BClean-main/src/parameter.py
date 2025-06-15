import Levenshtein
import numpy as np
import math
import re


class CPT_estimator:
	def __init__(self, estimator):
		self.estimator = estimator
	
	def get_cpts(self, model, data, choice):
		if (choice == "org"):
			model.fit(data, estimator = self.estimator)
			return model
		for sub_net in model:
			attrs_temp = list(model[sub_net].nodes)
			data_temp = data[attrs_temp]
			model[sub_net].fit(data_temp, estimator = self.estimator)
		return model


class CompensativeParameter:
	def __init__(self, attr_type, domain, occurrence, model, df):
		self.attr_type = attr_type
		self.domain = domain
		self.occurrence = occurrence
		self.model = model
		self.df = df
		self.tf_idf = {}
	
	def return_penalty(self, obs, attr, index, data_line, prior):
		p_obs_G_cand = {}
		domain_base = {}
		cooccurance_base = {}
		cooccurance_dist = {}
		total_ground = 0
		toatl_cooccur = 0
		occurrence = self.occurrence[attr]
		attr_type = self.attr_type
		
		if (obs == "A Null Cell" and attr_type[attr]["AllowNull"] == "N"):
			obs = ""
		for groud in prior:
			domain_base[groud] = 1 + Levenshtein.distance(obs, groud)
			total_ground += domain_base[groud]
			cooccurance_base_vec = []
			for other_attr in attr_type:
				if (other_attr == attr or other_attr in self.model.get_parents(
						node = attr) or other_attr in self.model.get_children(node = attr)):
					continue
				other_val = data_line.loc[index, other_attr]
				score = occurrence[groud][other_attr][other_val] if (other_val in occurrence[groud][other_attr]) else 0
				cooccurance_base_vec.append(score)
			
			# cooccurance_dist[groud] = np.linalg.norm(np.array(cooccurance_base_vec), ord=2) + 1
			cooccurance_dist[groud] = np.linalg.norm(np.array(cooccurance_base_vec), ord = 2) + math.exp(
				(-1) * Levenshtein.distance(groud, obs)) + 1
			toatl_cooccur += cooccurance_dist[groud]
		
		for groud in prior:
			if (not ((attr_type[attr]["AllowNull"] == "Y" or
			          (attr_type[attr]["AllowNull"] == "N" and
			           groud != "A Null Cell")) and
			         (attr_type[attr]["pattern"] == None or
			          (attr_type[attr]["pattern"] != None and
			           re.search(attr_type[attr]["pattern"], groud))))
			):
				domain_base[groud] = 0
				cooccurance_dist[groud] = 0
				p_obs_G_cand[groud] = 0
			else:
				domain_base[groud] = domain_base[groud] / total_ground
				cooccurance_dist[groud] = cooccurance_dist[groud] / toatl_cooccur
				p_obs_G_cand[groud] = cooccurance_dist[groud]
		
		return p_obs_G_cand
	
	def return_penalty_test(self, obs, attr, index, data_line, prior, attr_order):
		if (self.tf_idf[attr] == None):
			return {p: 1 for p in prior}
		p_obs_G_cand = {}
		obs_combine = ""
		combine_attrs, dic, dic_idf = self.tf_idf[attr]
		for at in combine_attrs:
			obs_combine += "," + data_line.loc[index, at]
		
		for ground in prior:
			obs_context = ground + obs_combine
			tf = dic.loc[obs_context] if (obs_context in dic) else 0
			if (tf == 0):
				continue
			idf = math.log(self.df.shape[0] / (dic_idf.loc[obs] + 1))
			if (idf == 0):
				continue
			p_obs_G_cand[ground] = tf * idf
		return p_obs_G_cand
	
	def init_tf_idf(self, attr_order):
		for attr in self.attr_type:
			combine_attrs = []
			for at in attr_order:
				if (at == attr or not (at in self.model.get_parents(
						node = attr) or at in self.model.get_children(node = attr))):
					continue
				combine_attrs.append(at)
			if (len(combine_attrs) == 0):
				self.tf_idf[attr] = None
				continue
			context_attr = "_".join(combine_attrs)
			context_attr += "_TempAttribute"
			df = self.df.copy()
			for at in combine_attrs:
				if (context_attr not in df.columns):
					df[context_attr] = df[at]
				else:
					df[context_attr] = df[context_attr] + "," + df[at]
			
			context_obs = attr + "_" + context_attr
			df[context_obs] = df[attr] + "," + df[context_attr]
			dic = df[context_obs].value_counts()
			dic_idf = df[attr].value_counts()
			self.tf_idf[attr] = [combine_attrs, dic, dic_idf]

