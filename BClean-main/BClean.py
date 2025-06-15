from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from BClean.src.parameter import CompensativeParameter
from BClean.src.structure import Compensative
from BClean.src.structure import BN_Structure
from BClean.src.parameter import CPT_estimator
from BClean.src.infer import Inference
from BClean.dataset import Dataset
import time
import numpy as np

class BayesianClean:
    def __init__(self,
                 dirty_df,
                 clean_df,
                 cpt_est = "Bayes", #["Bayes", "Likehood"]
                 infer_strategy = "PI",
                 tuple_prun = 1.0,
                 maxiter = 1,
                 num_worker = 32,
                 chunksize = 250,
                 model_path=None,
                 model_save_path=None,
                 attr_type=None,
                 fix_edge=None,
                 model_choice=None,
                 model_update = False
                 ):
        self.start_time = time.perf_counter()
        print("+++++++++data loading++++++++")
        self.attr_type = attr_type
        self.dataLoader = Dataset()
        self.dirty_data = dirty_df
        self.clean_data = clean_df
        self.cpt_est = cpt_est
        self.fix_edge = fix_edge
        self.model_path = model_path
        self.model_save_path = model_save_path
        self.model_choice = model_choice
        self.infer_strategy = infer_strategy
        self.tuple_prun = tuple_prun
        self.maxiter = maxiter
        self.num_worker = num_worker
        self.chunksize = chunksize
        self.model_update = model_update
        self.data = self.dataLoader.pre_process_data(self.dirty_data, self.attr_type)
        print("+++++++++computing error cell++++++++")
        self.actual_error = self.dataLoader.get_error(self.dirty_data, self.clean_data)
        print("error:{}".format(len(self.actual_error)))
        print("+++++++++error cell computing complete++++++++")
        
    def get_BN(self, data):
        structure_learing = BN_Structure(data = data,
                                         model_path = self.model_path,
                                         model_save_path = self.model_save_path,
                                         model_choice = self.model_choice,
                                         infer_strategy = self.infer_strategy,
                                         fix_edge = self.fix_edge)
        return structure_learing.get_bn()
    
    def BN_init_parameter(self, data, model, model_dict, cpt_estimator = None):
        if(self.cpt_est == "Bayes"):
            cpt_estimator = CPT_estimator(estimator = BayesianEstimator)
        elif(self.cpt_est == "Likehood"):
            cpt_estimator = CPT_estimator(estimator = MaximumLikelihoodEstimator)
        model = cpt_estimator.get_cpts(model = model, data = data, choice = "org")
        model_dict = cpt_estimator.get_cpts(model = model_dict, data = data, choice = "PI")
        
        return model, model_dict, cpt_estimator
    
    def init_Compensative(self, data, attr_type, model):
        compensative = Compensative(data, self.attr_type)
        
        Occurrence_list, Frequence_list, Occurrence_1 = compensative.build()
        
        compensativeParameter = CompensativeParameter(attr_type = attr_type,
                                                      domain = Frequence_list,
                                                      occurrence = Occurrence_list,
                                                      model = model,
                                                      df = data)
        
        return Occurrence_list, Frequence_list, Occurrence_1, compensativeParameter
    
    def clean_setting(self, data, model, model_dict, Frequence_list, Occurrence_1, CompensativeParameter):
        return Inference(dirty_data = self.dirty_data,
                         data = data,
                         model = model,
                         model_dict = model_dict,
                         attrs = self.attr_type,
                         Frequence_list = Frequence_list,
                         Occurrence_1 = Occurrence_1,
                         CompensativeParameter = CompensativeParameter,
                         infer_strategy = self.infer_strategy,
                         chunksize = self.chunksize,
                         num_worker = self.num_worker,
                         tuple_prun = self.tuple_prun
                         )
    
    def detection(self):
        pass
    
    def correction(self, inference):
        repair_list = inference.Repair(self.attr_type)
        repair_list["clean"] = self.clean_data.copy()
        return repair_list
    
    def clean(self):
        print("BN generation")
        model, model_dict = self.get_BN(data = self.data)
        print("picking up BN")
        model, model_dict, cpt_estimator = self.BN_init_parameter(data = self.data, model = model, model_dict = model_dict)
        print("compensative generation")
        Occurrence_list, Frequence_list, Occurrence_1, compensativeParameter = self.init_Compensative(data = self.data,
                                                                                                      attr_type = self.attr_type,
                                                                                                      model = model)
        repair_list = None
        print("cleaning")
        for iter in range(self.maxiter):
            infer = self.clean_setting(model = model,
                                       data = self.data,
                                       model_dict = model_dict,
                                       Frequence_list = Frequence_list,
                                       Occurrence_1 = Occurrence_1,
                                       CompensativeParameter = compensativeParameter)
            repair_list = self.correction(inference = infer)
            if (self.model_update == True):
                print("updating model and parameter")
                model, data, model_dict, Frequence_list, Occurrence_1, compensativeParameter = self.reconstruct(data_list = repair_list,
                                                                                                                cpt_estimator = cpt_estimator)
            print(print("++++++++++++++++The {}/{} Iteration+++++++++++++++++".format(iter + 1, self.maxiter)))
        return repair_list
        
    def transform_data(self, data):
        data = data.fillna("A Null Cell")
        all_domain_map = {}
        for val in data:
            domain_val = data[val].unique()
            count = 0
            domain_map = {}
            for v in domain_val:
                domain_map[v] = count
                count += 1
            data[val] = data[val].apply(lambda x : x)
            all_domain_map[val] = domain_map
        return data, all_domain_map

    def produce_train(self, data, attrs):
        attrs = [attr for attr in attrs]
        data = data[attrs]
        data = data.applymap(lambda x: np.NAN if(x == "A Null Cell") else x)
        return data
    
    def reconstruct(self, data_list, cpt_estimator):
        dirty_data, result = data_list["dirty"], data_list["result"]
        model, model_dict = self.get_BN(data = result)
        model, model_dict, cpt_estimator = self.BN_init_parameter(data = result,
                                                                  model = model,
                                                                  model_dict = model_dict,
                                                                  cpt_estimator = cpt_estimator)
        
        Occurrence_list, Frequence_list, Occurrence_1, compensativeParameter = self.init_Compensative(data = result,
                                                                                                      attr_type = self.attr_type,
                                                                                                      model = model)
        
        return model, result, model_dict, Frequence_list, Occurrence_1, compensativeParameter
    