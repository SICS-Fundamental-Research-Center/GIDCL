import cx_Oracle
import pandas as pd
from SeqGAN.get_config import get_config
config = get_config('config.ini')

def evaluate(path_ori,path, path_repair):
    # path_ori="Hosp_rules"
    # path="Hosp_rules_copy"
    # path_ori = "LETTER"
    # path = "LETTER_copy"\
    print("进行评估")
    df = pd.read_csv(path, encoding = "utf-8", dtype = str)
    clean_df = pd.read_csv(path_ori, encoding = "utf-8", dtype = str)
    repair_df = pd.read_csv(path_repair, encoding = "utf-8", dtype = str)
    
    total_err = 0
    rep = 0
    right_rep = 0
    err_rep = 0
    missing_rep = 0
    for ind in repair_df.index:
        for col in repair_df.columns:
            if(col == "Label"):
                continue
            if(str(repair_df.loc[ind, col]) != str(df.loc[ind, col])):
                rep += 1
                if(str(repair_df.loc[ind, col]) == str(clean_df.loc[ind, col])):
                    right_rep += 1
                else:
                    err_rep += 1
            if(str(df.loc[ind, col]) != str(clean_df.loc[ind, col])):
                total_err += 1
                if(str(repair_df.loc[ind, col]) != str(clean_df.loc[ind, col])):
                    missing_rep += 1
    precision = right_rep / rep
    recall = (total_err - missing_rep) / total_err

    
                

    #
    # # conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    # # cursor = conn.cursor()
    #
    # # sql1 = "select * from \"" + path + "\" where \"Label\"='2' or \"Label\"='3'"    #where rownum < 3  #order by "Provider ID" desc
    # # print(sql1)
    # # cursor.execute(sql1)
    # # data1 = cursor.fetchall()
    # data1 = df.loc[df["Label"] == "2" or df["Label"] == "3"]
    # # print(data1)
    # # des = cursor.description
    # des = list(df.columns)
    # att = []
    # for item in des:
    #     att.append(item[0])
    # print(att)
    # t2 = len(data1[0]) - 1  # 每行数据长度，-1是为了变成索引位置
    #
    #
    # #print("1")
    # # print(data1)
    # correct=0
    # error=0
    # len_update=len(data1.values.tolist())
    # for row in data1.values.tolist():
    #     # print(row)
    #     clean_temp = clean_df.copy()
    #     for i in range(t2):
    #         clean_temp = clean_temp.loc[clean_temp[att[i]] == row[i]]# t2
    #         # if i == 0:
    #         #     sql_info = "\"" + att[i] + "\"='" + row[i] + "'"
    #         # else:
    #         #     sql_info = sql_info + " and \"" + att[i] + "\"='" + row[i] + "'"
    #     # sql_info_ori=sql_info
    #     sql_info_label = sql_info + " and \"Label\"='2'"
    #     sql2 = "select * from \"" + path_ori + "\" where " + sql_info + ""  # where rownum < 3  #order by "Provider ID" desc
    #     print(sql2)
    #     cursor.execute(sql2)
    #     data_ori = cursor.fetchall()
    #     print(data_ori)
    #     if data_ori==[]:
    #         sql_update = "update \"" + path + "\" set \"Label\"='3'  where  " + sql_info + ""
    #         # print("原始：", sql_info)
    #         # print("Update信息：", sql_update)
    #         cursor.execute(sql_update)
    #         conn.commit()
    #         # print("目标结果缺失，原始正确数据中无与修复后相同的数据")
    #         # print(sql2)
    #         error += 1
    #         continue
    #     x1=data_ori[0]
    #     x1=list(x1)[:-1]
    #     # print("修复后数据",x1)
    #
    #     sql3 = "select * from \"" + path + "\" where " + sql_info_label + ""  # where rownum < 3  #order by "Provider ID" desc
    #     # print(sql3)
    #     cursor.execute(sql3)
    #     data_new = cursor.fetchall()
    #     # print(data_new)
    #     x2 = data_new[0]
    #     x2 = list(x2)[:-1]
    #     # print("修复后数据", x1)
    #     # print("正确数据  ",x2)
    #     if x1==x2:
    #         correct+=1
    #     else:
    #         print("修复错误，修复为", x1,"         实际正确数据为",x2)
    #         error+=1
    # precision=1-error/(error+correct)
    #
    # # print(correct,error,len_update)
    # # print("precision:",precision)
    #
    #
    # sql4 = "select * from \"" + path + "\" where \"Label\"='2' or \"Label\"='1' or  \"Label\"='3'"
    # # print(sql4)
    # cursor.execute(sql4)
    # data2 = cursor.fetchall()
    # recall = correct / len(data2)
    #
    # print(len(data2),"个错误，修复了",len_update,"个，其中正确数量为",correct,"，错误数量为",error)
    # # print(correct, error, len_update)
    # print("precision:", precision)
    # print("recall:", recall)
    # cursor.close()
    # conn.close()
    f_measure = 2 * precision * recall / (precision + recall)
    print(f_measure)
    print("total_err:{}, rep:{}, right_rep:{},err_rep:{} ,missing_rep:{}".format(total_err, rep, right_rep, err_rep, missing_rep))
    with open(config["path_eval"], 'a') as f:
        f.write("")
        f.write(str(total_err))
        f.write("个错误，修复了")
        f.write(str(rep))
        f.write("个，其中正确数量为")
        f.write(str(right_rep))
        f.write("，错误数量为")
        f.write(str(err_rep))
        f.write("precision:")
        f.write(str(precision))
        f.write("           recall:")
        f.write(str(recall))
        f.write("           f_measure:")
        f.write(str(f_measure))
        f.write("\n")

        f.close()


if __name__ == '__main__':

    # flag = 2
    # if flag == 1:
    #     path_ori = "Test"  # Hosp_rules
    #     path = "Test_copy"  #
    # if flag == 2:
    #     path_ori = "Hosp_rules"  #
    #     path = "Hosp_rules_copy"
    # if flag == 3:
    #     path_ori = "UIS"  #
    #     path = "UIS_copy"
    #
    # if flag == 4:
    #     path_ori = "Food"  #
    #     path = "Food_copy"

    evaluate("/home/hsf/code/Garf-master-main/Garf-master-main/data/Inpatient/Inpatient_clean.csv",
             "/home/hsf/code/Garf-master-main/Garf-master-main/data/Inpatient/Inpatient_dirty.csv",
             "/home/hsf/code/Garf-master-main/Garf-master-main/data/Inpatient/Inpatient_repair.csv")
