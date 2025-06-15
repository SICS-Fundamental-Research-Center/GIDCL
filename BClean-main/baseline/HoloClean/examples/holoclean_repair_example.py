import sys
sys.path.append('../')
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *
import time

# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    epochs=5         ,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=True,
    timeout=24*3600,
    feature_norm=False,
    weight_norm=False,
    print_fw=True
).session

# 2. Load training data and traints.
hc.load_data('soccer', '../testdata/Soccers_sample_dirty.csv')
# hc.load_dcs('../testdata/hospital_constraints.txt')
hc.load_dcs('../testdata/Soccer_constraints.txt')
start = time.clock()
hc.ds.set_constraints(hc.get_dcs())

# 3. Detect erroneous cells using these two detectors.
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)

# 4. Repair errors utilizing the defined features.
hc.setup_domain()
featurizers = [
    InitAttrFeaturizer(),
    OccurAttrFeaturizer(),
    FreqFeaturizer(),
    ConstraintFeaturizer(),
]

hc.repair_errors(featurizers)

# 5. Evaluate the correctness of the results.
hc.evaluate(fpath='../testdata/Soccers_sample_clean_holoclean.csv',
            tid_col='tid',
            attr_col='attribute',
            val_col='correct_val')
end = time.clock()
print("++++++time using:{:.3f}+++++++".format(end - start))
