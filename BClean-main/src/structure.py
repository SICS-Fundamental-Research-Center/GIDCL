from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import ParameterEstimator
import pandas as pd
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, TreeSearch, PC
from pgmpy.estimators import BDeuScore, K2Score, BicScore, StructureScore, BDsScore
from pgmpy.estimators import MmhcEstimator
import joblib
import Levenshtein

import re
import time
from sklearn.covariance import graphical_lasso
from sklearn import covariance, preprocessing
from scipy import sparse
from sksparse.cholmod import cholesky, analyze
import numpy as np
from tqdm import tqdm

class BN_Structure:
	def __init__(self, data, model_path, model_choice, infer_strategy, fix_edge, model_save_path = None):
		self.model = None
		self.model_dict = None
		self.fix_edge = fix_edge
		self.data = data
		self.model_choice = model_choice
		self.infer_strategy = infer_strategy
		self.model_path = model_path
		self.model_save_path = model_save_path
	
	def get_bn(self):
		'''
		:return: Original model and partition model
		'''
		df1 = self.data.copy()
		df2 = df1.copy()
		print(df2)
		if (self.model_path != None):
			G = joblib.load(self.model_path)
		else:
			hc = HillClimbSearch(df2, complete_samples_only = True)
			model = None
			
			if (self.model_choice == "appr"):
				start_appr = time.perf_counter()
				Edeges = self.get_rel(df2)
				model_appr = BayesianNetwork()
				for attr in list(df2.columns):
					model_appr.add_node(attr)
				for edge in Edeges:
					model_appr.add_edge(edge[0], edge[1])
				end_appr = time.perf_counter()
				model = model_appr
				print("fdx time_using:{}".format(end_appr - start_appr))
			
			elif (self.model_choice == "bdeu"):
				start_bdu = time.perf_counter()
				model_bdu = hc.estimate(scoring_method = BDeuScore(df2, equivalent_sample_size = 5),
				                        tabu_length = 10,
				                        epsilon = 1e-4,
				                        max_iter = 20,
				                        fixed_edges = self.fix_edge,
				                        max_indegree = 2)
				end_bdu = time.perf_counter()
				model = model_bdu
				print("bdu time_using:{}".format(end_bdu - start_bdu))
			
			elif (self.model_choice == "bic"):
				start_bic = time.perf_counter()
				model_bic = hc.estimate(scoring_method = BicScore(df2),
				                        tabu_length = 10,
				                        epsilon = 1e-4,
				                        max_iter = 20,
				                        fixed_edges = self.fix_edge,
				                        max_indegree = 2)
				end_bic = time.perf_counter()
				model = model_bic
				print("bic time_using:{}".format(end_bic - start_bic))
			
			elif (self.model_choice == "k2"):
				start_k2 = time.perf_counter()
				model_k2 = hc.estimate(scoring_method = K2Score(df2),
				                       tabu_length = 10,
				                       epsilon = 1e-4,
				                       max_iter = 20,
				                       fixed_edges = self.fix_edge,
				                       max_indegree = 2)
				end_k2 = time.perf_counter()
				model = model_k2
				print("k2 time_using:{}".format(end_k2 - start_k2))
			
			elif (self.model_choice == "bds"):
				start_bds = time.perf_counter()
				model_bds = hc.estimate(scoring_method = BDsScore(df2, equivalent_sample_size = 5),
				                        tabu_length = 10,
				                        epsilon = 1e-4,
				                        max_iter = 20,
				                        fixed_edges = self.fix_edge,
				                        max_indegree = 2)
				end_bds = time.perf_counter()
				model = model_bds
				print("bds time_using:{}".format(end_bds - start_bds))
			
			elif (self.model_choice == "fix"):
				model = BayesianNetwork()
				for node in df2.columns:
					model.add_node(node)
				for edge in self.fix_edge:
					model.add_edge(edge[0], edge[1])
			
			G = BayesianNetwork()
			for node in model.nodes:
				G.add_node(node)
			for edge in model.edges:
				G.add_edge(edge[0], edge[1])
		model_dict = {}
		
		for key in tqdm(G.nodes):
			parend_nodes = G.get_parents(node = key)
			kids_nodes = G.get_children(node = key)
			model_temp = BayesianNetwork()
			temp_attrs = parend_nodes + kids_nodes + [key]
			model_temp.add_node(key)
			for node in parend_nodes:
				model_temp.add_node(node)
				model_temp.add_edge(node, key)
			for node in kids_nodes:
				model_temp.add_node(node)
				model_temp.add_edge(key, node)
			data_temp = df2[temp_attrs].copy()
			print(len(temp_attrs), temp_attrs)
			model_temp.fit(data_temp, estimator = BayesianEstimator)
			model_dict[key] = model_temp
		print(G.nodes())
		# print(G.cpds)
		print(G.edges())
		if (self.model_save_path is not None):
			joblib.dump(G, self.model_save_path)
			print("++++++model saved++++++")
		self.model = G
		self.model_dict = model_dict
		return self.model, self.model_dict
	
	def get_rel(self, data):
		'''
		:param data: Observation
		:return: Aprroximate edges of bn
		'''
		attr = list(data.columns)
		np.random.shuffle(attr)
		data = data.copy()
		data = data.loc[:, attr]
		values = data.values.tolist()
		line, row = len(values), len(values[1])
		threshold = 1
		output = [[0 for i in range(row)] for i in range(row * line * threshold)]
		for i in tqdm(range(row)):
			Di = list(sorted(values, key = lambda x: str(x[i])))
			Di_shfit = list(Di)
			for k in range(threshold):
				Di_shfit = list(Di_shfit)
				Di_shfit_turnover = Di_shfit[len(Di_shfit) - 1]
				Di_shfit = list(Di_shfit[: len(Di_shfit) - 1])
				Di_shfit.insert(0, Di_shfit_turnover)
				for j in range(line):
					for l in range(row):
						output[i * (line * threshold) + k * line + j][l] = 1 - (
									2 * Levenshtein.distance(Di[j][l], Di_shfit[j][l])
									/ (len(Di[j][l]) + len(Di_shfit[j][l])))
		
		one_zero_matrix = output
		myScaler = preprocessing.StandardScaler()
		X = myScaler.fit_transform(one_zero_matrix)
		Y = covariance.empirical_covariance(X)
		# shrunk_cov = covariance.shrunk_covariance(Y, shrinkage=0.5)
		est_cov, inv_cov = graphical_lasso(Y, alpha = 0.001, mode = 'cd', max_iter = 20000)
		
		B_dataframe_relation = pd.DataFrame(est_cov, columns = attr, index = attr)
		
		I = np.eye(inv_cov.shape[0])
		P = np.rot90(I)
		P_x_inv_cov = np.dot(P, inv_cov)
		P_transpose = P.transpose()
		PAP = np.dot(P_x_inv_cov, P_transpose)
		PAP = sparse.csc_matrix(PAP)
		factor = cholesky(PAP)
		L = factor.L_D()[0].toarray()
		U = np.dot(np.dot(P, L), P.transpose())
		B = I - U
		# print(B)
		B_dataframe = pd.DataFrame(B, columns = attr, index = attr)

		Edges = []
		
		for i in range(len(B)):
			for j in range(len(B[0])):
				if (B[i][j] > 0.3):
					Edges.append((attr[i], attr[j]))
		
		B_dataframe_relation = pd.DataFrame(est_cov, columns = attr, index = attr)
		
		print("edges discovery:{}".format(Edges))
		return Edges

class Compensative:
	def __init__(self, data, attrs_type):
		self.data = data
		self.attrs_type = attrs_type
		self.Occurrence_list = {}
		self.Frequence_list = {}
		self.Occurrence_1 = {}
		
	def build(self):
		self.occur_and_fre(self.data, self.attrs_type)
		return self.Occurrence_list, self.Frequence_list, self.Occurrence_1
	
	def occur_and_fre(self, data, attrs):
		'''
		:param attrs: UC
		:return: Parameter of Compensative dictionary
		'''
		self.Occurrence_list, self.Frequence_list, self.Occurrence_1 = {}, {}, {}
		df = data.copy()
		
		for attr in df.columns:
			val_count = df[attr].value_counts()
			self.Frequence_list[attr] = {}
			for val in val_count.index:
				self.Frequence_list[attr][val] = val_count[val]
				
		indexs = [(i, attr, list(df.columns), attrs) for i in range(df.shape[0]) for attr in list(df.columns)]
		
		for index in tqdm(indexs):
			self._corrlate(index)
	
	def _corrlate(self, args):
		t_id, attr_main, attrs, data_prior = args
		weight = len(attrs) ** 2
		pen_weight = weight
		confident = 1
		if (attr_main not in self.Occurrence_list):
			self.Occurrence_list[attr_main] = {}
			self.Occurrence_1[attr_main] = {}
		main_val = self.data.loc[t_id, attr_main]
		if (not ((data_prior[attr_main]["AllowNull"] == "Y" or
		          (data_prior[attr_main]["AllowNull"] == "N" and
		           main_val != "A Null Cell")) and
		         (data_prior[attr_main]["pattern"] == None or
		          (data_prior[attr_main]["pattern"] != None and
		           re.search(data_prior[attr_main]["pattern"], main_val))))
		):
			pen_weight -= 2 * weight * weight
			confident = 0
		if (main_val not in self.Occurrence_list[attr_main]):
			self.Occurrence_list[attr_main][main_val] = {}
			self.Occurrence_1[attr_main][main_val] = {}
		
		for attr_vice in attrs:
			if (attr_main == attr_vice):
				continue
			if (attr_vice not in self.Occurrence_list[attr_main][main_val]):
				self.Occurrence_list[attr_main][main_val][attr_vice] = {}
				self.Occurrence_1[attr_main][main_val][attr_vice] = {}
			vice_val = self.data.loc[t_id, attr_vice]
			if (not ((data_prior[attr_vice]["AllowNull"] == "Y" or
			          (data_prior[attr_vice]["AllowNull"] == "N" and
			           vice_val != "A Null Cell")) and
			         (data_prior[attr_vice]["pattern"] == None or
			          (data_prior[attr_vice]["pattern"] != None and
			           re.search(data_prior[attr_vice]["pattern"], vice_val))))
			):
				confident *= 0.5
				pen_weight -= 2 * weight
			if (vice_val not in self.Occurrence_list[attr_main][main_val][attr_vice]):
				self.Occurrence_list[attr_main][main_val][attr_vice][vice_val] = 0
				self.Occurrence_1[attr_main][main_val][attr_vice][vice_val] = 0
		
		for attr_vice in attrs:
			if (attr_main == attr_vice):
				continue
			vice_val = self.data.loc[t_id, attr_vice]
			self.Occurrence_1[attr_main][main_val][attr_vice][vice_val] += 1
			if (confident >= 0.5):
				self.Occurrence_list[attr_main][main_val][attr_vice][vice_val] += weight
			elif (confident == 0):
				self.Occurrence_list[attr_main][main_val][attr_vice][vice_val] = 0
			else:
				self.Occurrence_list[attr_main][main_val][attr_vice][vice_val] += pen_weight
			self.Occurrence_list[attr_main][main_val][attr_vice][vice_val] = max(
				self.Occurrence_list[attr_main][main_val][attr_vice][vice_val], 0)