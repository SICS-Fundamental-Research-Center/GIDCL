import time

import pandas as pd
import tqdm
from concurrent import futures
import multiprocessing

def _transformdata(args):
    tid, df = args
    res = pd.DataFrame(columns=["tid", "attribute", "correct_val"])
    for attr in list(df.columns):

        data = [[tid, attr, df.loc[tid, attr]]]
        dataline = pd.DataFrame(data, columns=["tid", "attribute", "correct_val"])
        res = res.append(dataline, ignore_index=True)
        # dataline.loc[0, "tid"] = tid
        # dataline.loc[0, "attribute"] = attribute
        # dataline.loc[0, "correct_val"] = df.loc[index, attribute]
    return {"cell": tid, "data": res}

if __name__ == '__main__':
    df1 = pd.read_csv("Soccer_clean.csv")
    result = pd.DataFrame(columns=["tid", "attribute", "correct_val"])
    for tid in tqdm.tqdm(range(df1.shape[0])):
        for attribute in list(df1.columns):
            data = [[tid, attribute, df1.loc[tid, attribute]]]
            dataline = pd.DataFrame(data, columns=["tid", "attribute", "correct_val"])
            result = result.append(dataline, ignore_index=True)
    # indexs = [i for i in range(df1.shape[0])]
    # executer = futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
    # fs = []
    # for index in tqdm.tqdm(indexs):
    #     r = executer.submit(_transformdata, [index, df1])
    #     fs.append(r)
    # # time.sleep()
    # while (True):
    #     flag = True
    #     res = []
    #     for f in fs:
    #         flag = True and f.done()
    #         if (f.done()):
    #             res.append(f.result())
    #             fs.pop(fs.index(f))
    #         # if(flag == False):
    #         #     break
    #     if (flag == True):
    #         executer.shutdown()
    #         break
    # print("complete")
    # res = list(sorted(res, key=lambda x: x["cell"]))
    #
    # for r in res:
    #     result = result.append(r["data"], ignore_index=True)
    #     print("++++++++++++{} cell collect++++++++++++".format(result.shape[0]))
    # # for index in tqdm.tqdm(range(inp.shape[0])):
    # #     for attr in list(inp.columns):
    # #         dataline = pd.DataFrame(columns=["tid", "attribute", "correct_val"])
    # #         dataline.loc[0, "tid"] = index
    # #         dataline.loc[0, "attribute"] = attr
    # #         dataline.loc[0, "correct_val"] = inp.loc[index, attr]
    # #         res = res.append(dataline, ignore_index=True)
    print("complete")
    result.to_csv("Soccer_clean1.csv")