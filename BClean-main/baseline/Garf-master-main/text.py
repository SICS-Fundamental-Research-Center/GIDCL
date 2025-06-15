import pandas as pd
import os
import math

def filter(path):
    print("执行了filter")
    print(os.getcwd())
    f = open('data/save/att_name.txt', 'r')
    label2att = eval(f.read())
    f.close()
    att2label = {v: k for k, v in label2att.items()}  # 字典反向传递
    f = open('data/save/rules_final.txt', 'r')
    rules_final = eval(f.read())
    f.close()
    l1 = len(rules_final)
    # print(rules_final)
    num = 0
    
    # conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    # cursor = conn.cursor()
    
    df = pd.read_csv(path, encoding = "utf-8", dtype = str)
    
    for rulename, ruleinfo in list(rules_final.items()):
        rows = df.copy()
        num += 1
        # print("过滤第", num, "条规则及对应数据")
        # print("ruleinfo:", ruleinfo)
        
        left = list(ruleinfo['reason'].keys())
        # print(left)
        word = list(ruleinfo['reason'].values())
        # print(word)
        k = list(ruleinfo['result'].keys())
        right = k[0]
        v = list(ruleinfo['result'].values())
        result = v[0]
        
        for i in range(len(left)):
            rows = rows.loc[rows[left[i]] == word[i]]
            
        # sqlex = left[0] + "\"='" + word[0] + "'"
        # i = 1
        # while (i < len(left)):
        # 	sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
        # 	i += 1
        
        # sql1 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex
        # # print(sql1)         #select "MINIT" from "UIS_copy" where "CUID"='9078' and "RUID"='15896' and "SSN"='463210223' and "FNAME"='Monken'
        
        rows = rows[left]
        
        # cursor.execute(sql1)  # "City","State" ,where rownum<=10
        # rows = cursor.fetchall()
        
        # rows = df.loc[]
        num1 = len(rows)
        if num1 < 3:
            # print("满足规则的数据有",num1,"条，推测来源为错误数据，无修复意义，删除规则",rules_final[str(rulename)])
            del rules_final[str(rulename)]
            continue
        else:
            t_rule = 1
            for row in rows:
                if (str(row[-1]) == str(result)):  # 此时规则与数据相符合, 规则置信度增加
                    t_rule = t_rule + 1
                    print("-->", t_rule, end = '')
                else:  # 此时规则与数据相违背, 规则置信度减少
                    t_rule = t_rule - 2
                    print("-->", t_rule, end = '')
                    flag = 0  # 标记该规则与数据存在冲突
            rules_final[str(rulename)].update({'confidence': t_rule})  # 规则置信度初始化
        # rules_final[str(rulename)].update({'confidence': 1})
    # sql2 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex + " and \"" + right + "\"='" + result + "'"
    # # print(sql2)
    # cursor.execute(sql2)
    # rows = cursor.fetchall()
    # num2 = len(rows)
    # # print(num2)
    # ratio = num2 / num1
    # if ratio < 0.51:
    #     # print("推测来源为错误数据，无修复意义，删除规则")
    #     del rules_final[str(rulename)]
    #     continue

    # cursor.close()
    # conn.close()
    
    f = open('data/save/rules_final.txt', 'w')
    f.write(str(rules_final))
    f.close()
    l2 = len(rules_final)
    print("规则过滤完成，剩余数量为", )
    print(str(l2))
    # print(str(rules_final))           #过滤后规则
    with open('data/save/log_filter.txt', 'w') as f:
        f.write("原始规则数量为")
        f.write(str(l1))
        f.write("规则过滤后，剩余数量为")
        f.write(str(l2))
        f.write("__________")
    f.close()

def addtwodimdict(thedict, key_a, key_b, val):
  if key_a in thedict:
    thedict[key_a].update({key_b: val})
  else:
    thedict.update({key_a:{key_b: val}})

def detect(rows,result,rulename,LHS,RHS,att2label,label2att):
    
    f = open('data/save/att_name.txt', 'r')
    label2att = eval(f.read())
    f.close()
    # print(label2att)
    att2label = {v: k for k, v in label2att.items()}  # 字典反向传递
    f = open('data/save/rules_final.txt', 'r')
    rule = eval(f.read())
    
    dert = 0
    t0=1
    t_rule=t0
    t_tuple=t0
    t_max=t_tuple   #满足rule条件的不同tuple中置信度最大值
    flag=1         #标记该规则是否与数据存在冲突
    flag_trust = 0  # 0代表相信数据，1代表相信规则
    for row in rows:
        if (str(row[RHS]) == str(result)):
            continue
        else:
            dert += 1
            flag = 0   # 标记该规则与数据存在冲突
    if (flag==1):           #该规则不与数据存在冲突,则直接给一个极大置信度
        t_rule=t_rule+100
        flag_trust = 3  # 3代表规则正确, 且无任何冲突
        return flag_trust
    else:                   #该规则与数据存在冲突,则计算每个tuple的置信度,以调整t_rule
        print("该规则与数据存在冲突")
        print("本次修复预计变化量", dert)
        error_row=[]
        rule_other=[]
        t_rule=t0
        for row in rows:    #每个满足规则条件的tuple
            AV_p=[]
            t_tp = 999   #当前tuple的置信度, 计算为一个tuple中不同AV_i中的置信度最小值,为了避免初始值干扰,先设定个大值
            t_tc = t0
            # flag_p=0     #用以记录AV_p中置信度最小的对应的属性位置
            # rule_p_name=[] #用以记录能够修复上述的AV_p中置信度最小的对应的属性的具有最大置信度的规则
            # print("匹配当前规则的tuple为：", row)
            for i in LHS:       #计算一个tuple中不同AV_i中的置信度最小值
                AV_p.append(row[i])
                t_AV_i = t0
                # rulename_p_max = []
                # t_rmax = 0
                attribute_p=label2att[i]
                for rulename_p, ruleinfo_p in list(rule.items()):      #遍历字典
                    if rulename == rulename_p:
                        continue
                    if t_AV_i>100 or t_AV_i<-100:
                        break
                    v = list(ruleinfo_p['result'].values())
                    left = list(ruleinfo_p['reason'].keys())
                    word = list(ruleinfo_p['reason'].values())
                    k = list(ruleinfo_p['result'].keys())
                    t_r = ruleinfo_p['confidence']
                    if t_r<0:
                        continue
                    right = k[0]
                    if attribute_p == right:
                        flag_equal = 0  # 规则能否确定row[i]的标记
                        for k in range(len(left)):
                            if row[att2label[left[k]]] == word[k]:  # 若row[i]所在的tuple满足某条规则的全部AV_p,标记为1
                                flag_equal = 1
                            else:
                                flag_equal = 0
                                break
                        if flag_equal == 1:  # 若该tuple中row[i]能够被其他规则确定,检测其是否满足规则
                            # print(row, "中")
                            # print(right, "可以由其他规则确定：", ruleinfo)
                            result2 = v[0]
                            # if t_rmax < t_r:  # 记录这些规则中最大的规则置信度
                            #     t_rmax = t_rmax
                            #     rulename_p_max = rulename_p  # 记录该最可信规则在字典中的标识
                            if str(row[i]) == str(result2):    # 检索其他规则以确定该tuple中每个Token的置信度,满足则增加,反之则减
                                t_AV_i = t_AV_i + t_r
                            else:
                                t_AV_i = t_AV_i - t_r
                                print("匹配当前规则的tuple为：", row)
                                print("AV_p中",str(row[i]), "与", str(result2), "不符,对应的规则为", ruleinfo_p, "其置信度为", t_r)

                if t_tp > t_AV_i:
                    t_tp = t_AV_i
                    # flag_p=i
                    # rule_p_name=rulename_p_max


            for rulename_c, ruleinfo_c in list(rule.items()):  # 遍历字典,计算t_c
                if rulename==rulename_c:
                        continue
                v = list(ruleinfo_c['result'].values())
                left = list(ruleinfo_c['reason'].keys())
                word = list(ruleinfo_c['reason'].values())
                k = list(ruleinfo_c['result'].keys())
                t_r = ruleinfo_c['confidence']
                if t_r < 0:
                    continue
                right = k[0]
                attribute_c = label2att[RHS]
                if attribute_c == right:
                    flag_equal = 0  # 规则能否确定row[i]的标记
                    for k in range(len(left)):
                        if row[att2label[left[k]]] == word[k]:  # 若AV_c所在的tuple满足某条规则的全部AV_p,标记为1
                            flag_equal = 1
                        else:
                            flag_equal = 0
                            break
                    if flag_equal == 1:  # 若该tuple中AV_c能够被其他规则确定,检测其是否满足规则
                        result2 = v[0]
                        if str(row[RHS]) == str(result2):
                            t_tc = t_tc + t_r
                        else:
                            t_tc = t_tc - t_r
                            print("匹配当前规则的tuple为：", row)
                            print("AV_c中",str(row[RHS]), "与", str(result2), "不符,对应的规则为", ruleinfo_c, "其置信度为", t_r)

            if t_tp==999:        #说明其中所有单元都无法被其他规则确定, 将其值重置为t0
                t_tp=t0
            if t_tc < t_tp:
                t_tuple = t_tc
            else:
                t_tuple = t_tp


            # print("匹配该规则的部分为", AV_p, "-->",row[RHS],"其置信度为",t_tuple)
            if (str(row[RHS]) == str(result)):  # 该元组数据与规则相符合, 置信度增加
                # print("此时t_rule=",t_rule,"t_tuple=",t_tuple,"math.ceil(math.log(1+t_tuple))=",math.ceil(math.log(1+t_tuple)))
                # print("规则确定值为",result,";实际值为",row[RHS],"相符,规则置信度增加",t_rule, end='')
                if t_tuple>0:
                    t_rule = t_rule + math.ceil(math.log(1+t_tuple))
                else:
                    t_rule = t_rule + t_tuple
                t_max = t_max
                print("-->", t_rule, end='')
            else:  # 该元组数据与规则相违背, 计算对应tuple的置信度
                # print("此时t_rule=", t_rule, "t_tuple=", t_tuple, "int(math.log(abs(t_tuple)))=",
                #       int(math.log(abs(t_tuple))))
                # print("规则确定值为", result, ";实际值为", row[RHS], "违反,规则置信度降低", t_rule, end='')
                if t_tuple>0:
                    t_rule = t_rule - 2*t_tuple
                else:
                    t_rule = t_rule + math.ceil(math.log(1+abs(t_tuple)))
                    print("-->", t_rule, end='')
                    
                if (t_rule < -100):
                    flag_trust = 0
                    return flag_trust  # 此时规则置信度过小,直接跳出循环,标记为错误
            if t_max < t_tuple:
                t_max = t_tuple
            # if t_rule < t_max:
            #     flag_trust = 0
            #     return flag_trust
            # elif t_rule > t_max:
            #     error_row.append(row)
            #     rule_other.append(rule_p_name)


        print("最终规则置信度为",t_rule,"与其冲突的元组中置信度最大的为",t_max)
    if (t_rule > t_max ):
        flag_trust = 1  # 此时认为规则正确，修改数据
    elif (t_rule < t_max ):
        flag_trust = 0
            # print("最终规则置信度为", t_rule, "与其冲突的元组中置信度最大的为", t_max)
        return flag_trust  # 此时认为数据正确，修改规则
    rule[str(rulename)].update({'confidence': t_rule}) #规则置信度初始化可以考虑单拿出来
    print()
    return flag_trust
    
def repair(path):
    print("执行了repair")
    f = open('data/save/att_name.txt', 'r')
    label2att = eval(f.read())
    f.close()
    # print(label2att)
    att2label = {v: k for k, v in label2att.items()}  # 字典反向传递
    f = open('data/save/rules_final.txt', 'r')
    rule = eval(f.read())
    f.close()
    # print(rule)
    num = 0
    error_rule = 0
    error_data = 0
    
    df = pd.read_csv(path, encoding = "utf-8", dtype = str)
    
    # conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    # cursor = conn.cursor()
    
    for rulename, ruleinfo in list(rule.items()):
        num += 1
        print("修复第", num, "条规则及对应数据")
        # print("rulename:" + rulename)
        print("ruleinfo:", ruleinfo)
        
        left = list(ruleinfo['reason'].keys())
        # print(left)
        word = list(ruleinfo['reason'].values())
        # print(word)
        k = list(ruleinfo['result'].keys())
        right = k[0]
        v = list(ruleinfo['result'].values())
        result = v[0]
        
        LHS = []
        LHS.append(att2label[left[0]])
        RHS = att2label[right]
        
        sqlex = left[0] + "\"='" + word[0] + "'"
        
        rows = df.loc[df[left[0]] == word[0]].copy()
        
        i = 1
        # AV_p = "\""+left[0]+"\""+","     #把left里的数据转化为字符串形式
        while (i < len(left)):
            rows = rows.loc[rows[left[i]] == word[i]]
            # sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
            # AV_p = AV_p +"\""+ left[i]+"\""+","
            LHS.append(att2label[left[i]])
            i += 1
        # print(sqlex)
        # print("AV_p索引：",LHS,"AV_c索引：",RHS)
        # AV_c = "\"" + right + "\""
        # print("AV_p:",AV_p,"AV_c:",AV_c)
        
        # sql1 = "select * from \"" + path + "\" where \"" + sqlex
        # sql1 = "select " +AV_p+ AV_c + " from \"" + path + "\" where \"" + sqlex
        # sql1 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex
        # print(sql1)
        # cursor.execute(sql1)  # "City","State" ,where rownum<=10
        # rows = cursor.fetchall()
        # print("rows:")
        
        flag_trust = detect(rows, result, rulename, LHS, RHS, att2label, label2att)
        
        if (flag_trust == 3):  # 3代表规则正确, 且无任何冲突, 直接进行下一条规则
            continue
        
        if (flag_trust == 0):
            error_rule += 1
        
        s1 = 0
        while (flag_trust == 0 and s1 < 3):
            print("规则不可信，修复规则")
            print("修复规则右侧")
            s1 += 1
            result = self.multipredict_rules_probability(word)
            print("右侧更改为", result)
            flag_trust = self.detect(rows, result, rulename, LHS, RHS, att2label, label2att)
            
            # print("trust=",trust)
            if (flag_trust == 1):
                print("规则修复成功")
                addtwodimdict(rule, str(rulename), 'result', {str(right): str(result)})
                print("修改后规则为", rule[str(rulename)])
            elif (flag_trust == 0 and s1 == 5):
                print("规则右侧无可替换修复")
        
        s2 = 0
        while (flag_trust == 0 and s2 < 3):
            result = v[0]
            print("修复规则左侧")
            s2 += 1
            min = 10
            flag = int(att2label[left[0]])
            # print(flag)
            if (min > flag):
                min = flag  # 目前reason部分最左侧对应的索引
            # print(min)
            if (min == 0):
                print("规则左侧无可增加修复，删除该条规则")
                del rule[str(rulename)]
                break
            left_new = label2att[min - 1]
            print("增加", left_new, "信息")
            sqladd = "select \"" + left_new + "\" from \"" + path + "\" where \"" + sqlex + "and \"" + right + "\"='" + result + "'"
            print("sqladd:", sqladd)
            cursor.execute(sqladd)
            rows_left = cursor.fetchall()
            # print(rows[0][0])
            # print(word)
            # print(rule[str(rulename)])
            
            row
            
            # 重构字典
            if (rows_left == []):
                # print("规则左侧无满足条件修改,删除该条规则")
                del rule[str(rulename)]
                break
            # print(rows_left)
            addtwodimdict(rule[str(rulename)], 'reason', str(left_new), str(rows_left[0][0]))
            for n in range(len(word)):
                del rule[str(rulename)]['reason'][left[n]]
                addtwodimdict(rule[str(rulename)], 'reason', str(left[n]), str(word[n]))
            # left = list(ruleinfo['reason'].keys())
            ######否则，字典里新增的内容应该在最前面，但现在在最后面
            # tex=[]
            # tex.append(rows_left[0][0])
            # for t in range(len(word)):
            #     tex.append(word[t])
            # # print(tex)
            # word = tex
            left = list(ruleinfo['reason'].keys())
            word = list(ruleinfo['reason'].values())
            # print(word)
            # print(rule[str(rulename)])
            sqlex = left[0] + "\"='" + word[0] + "'"
            i = 1
            while (i < len(left)):
                sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
                i += 1
            sql1 = "select * from \"" + path + "\" where \"" + sqlex
            # print(sql1)
            cursor.execute(sql1)  # "City","State" ,where rownum<=10
            rows = cursor.fetchall()
            
            if (len(rows) < 3):
                continue
            
            result = self.multipredict_rules_argmax(word)
            # print(result)
            flag_trust = self.detect(rows, result, rulename, LHS, RHS, att2label, label2att)
            if (flag_trust == 1):
                print("规则修复成功")
                print("修改后规则为", rule[str(rulename)])
            elif (flag_trust == 1 and min != 0):
                # print("规则左侧无满足条件修改,删除该条规则")
                del rule[str(rulename)]
                break
        if (flag_trust == 0):
            print("规则无可用修复,删除该规则")
        
        if (flag_trust == 1):
            t0 = 1
            for row in rows:
                if (str(row[RHS]) == str(result)):
                    continue
                else:
                    AV_p = []
                    t_tp = 999  # 当前tuple的置信度, 计算为一个tuple中不同AV_i中的置信度最小值,为了避免初始值干扰,先设定个大值
                    t_tc = t0
                    flag_p = 0  # 用以记录AV_p中置信度最小的对应的属性位置
                    rule_p_name = []  # 用以记录能够修复上述的AV_p中置信度最小的对应的属性的具有最大置信度的规则
                    print("匹配当前规则的tuple为：", row)
                    for i in LHS:  # 计算一个tuple中不同AV_i中的置信度最小值
                        AV_p.append(row[i])
                        t_AV_i = t0
                        attribute_p = label2att[i]
                        rulename_p_max = []
                        t_rmax = -999  # 下面遍历的字典中能纠正AV_i的规则中最大置信度, 初始设为极小值
                        for rulename_p, ruleinfo_p in list(rule.items()):  # 遍历字典
                            if rulename == rulename_p:
                                continue
                            if t_AV_i > 100 or t_AV_i < -100:
                                break
                            v = list(ruleinfo_p['result'].values())
                            left = list(ruleinfo_p['reason'].keys())
                            word = list(ruleinfo_p['reason'].values())
                            k = list(ruleinfo_p['result'].keys())
                            t_r = ruleinfo_p['confidence']
                            if t_r < 0:
                                continue
                            right = k[0]
                            if attribute_p == right:
                                flag_equal = 0  # 规则能否确定row[i]的标记
                                for k in range(len(left)):
                                    if row[att2label[left[k]]] == word[k]:  # 若row[i]所在的tuple满足某条规则的全部AV_p,标记为1
                                        flag_equal = 1
                                    else:
                                        flag_equal = 0
                                        break
                                if flag_equal == 1:  # 若该tuple中row[i]能够被其他规则确定,检测其是否满足规则
                                    # print(row, "中")
                                    # print(right, "可以由其他规则确定：", ruleinfo)
                                    result2 = v[0]
                                    if t_rmax < t_r:  # 记录这些规则中最大的规则置信度
                                        t_rmax = t_rmax
                                        rulename_p_max = rulename_p  # 记录该最可信规则在字典中的标识
                                    if str(row[i]) == str(result2):
                                        t_AV_i = t_AV_i + t_r
                                    else:
                                        t_AV_i = t_AV_i - t_r
                                        print("AV_p中", str(row[i]), "与", str(result2), "不符,对应的规则为", ruleinfo_p,
                                              "其置信度为", t_r)
                        
                        if t_tp > t_AV_i:
                            t_tp = t_AV_i
                            flag_p = i  # 记录置信度最小的AV_i的索引
                            rule_p_name = rulename_p_max  # 记录能纠正该AV_i的置信度最大的规则名
                    
                    for rulename_c, ruleinfo_c in list(rule.items()):  # 遍历字典,计算t_c
                        if rulename == rulename_c:
                            continue
                        v = list(ruleinfo_c['result'].values())
                        left = list(ruleinfo_c['reason'].keys())
                        word = list(ruleinfo_c['reason'].values())
                        k = list(ruleinfo_c['result'].keys())
                        t_r = ruleinfo_c['confidence']
                        if t_r < 0:
                            continue
                        right = k[0]
                        attribute_c = label2att[RHS]
                        if attribute_c == right:
                            flag_equal = 0  # 规则能否确定row[i]的标记
                            for k in range(len(left)):
                                if row[att2label[left[k]]] == word[k]:  # 若AV_c所在的tuple满足某条规则的全部AV_p,标记为1
                                    flag_equal = 1
                                else:
                                    flag_equal = 0
                                    break
                            if flag_equal == 1:  # 若该tuple中AV_c能够被其他规则确定,检测其是否满足规则
                                result2 = v[0]
                                if str(row[RHS]) == str(result2):
                                    t_tc = t_tc + t_r
                                else:
                                    t_tc = t_tc - t_r
                                    print("AV_c中", str(row[RHS]), "与", str(result2), "不符,对应的规则为", ruleinfo_c, "其置信度为",
                                          t_r)
                    
                    if t_tp == 999:  # 说明其中所有单元都无法被其他规则确定, 将其值重置为t0
                        t_tp = t0
                    if t_tc < t_tp or t_tc == t_tp:
                        print("此时认为数据结果部分错误,根据规则修复数据,当前规则为", rulename, "-->", result, "t_p为", t_tp, "t_c为", t_tc)
                        for x in range(len(row) - 1):  # t2
                            if x == 0:
                                sql_info = "\"" + label2att[x] + "\"='" + row[x] + "'"
                            else:
                                sql_info = sql_info + " and \"" + label2att[x] + "\"='" + row[x] + "'"
                        sql_update = "update \"" + path + "\" set \"Label\"='2' , \"" + label2att[
                            RHS] + "\"='" + result + "' where  " + sql_info + ""
                        print("原始：", sql_info)
                        print("Update信息：", sql_update)
                        cursor.execute(sql_update)
                        conn.commit()
                    else:
                        print(rule_p_name)
                        if rule_p_name == []:
                            print("可能有错误")
                            continue
                        rname = rule[str(rule_p_name)]
                        v2 = list(rname['result'].values())
                        result2 = v2[0]
                        print("此时认为数据推论部分错误,根据规则修复数据,当前规则为", rule_p_name, "-->", result2, "t_p为", t_tp, "t_c为", t_tc)
                        for x in range(len(row) - 1):  # t2
                            if x == 0:
                                sql_info = "\"" + label2att[x] + "\"='" + row[x] + "'"
                            else:
                                sql_info = sql_info + " and \"" + label2att[x] + "\"='" + row[x] + "'"
                        sql_update = "update \"" + path + "\" set \"Label\"='2' , \"" + label2att[
                            flag_p] + "\"='" + result2 + "' where  " + sql_info + ""
                        print("原始：", sql_info)
                        print("Update信息：", sql_update)
                        cursor.execute(sql_update)
                        conn.commit()
                        continue
    
    # if (flag_trust == 1):
    #     # 只修复标记为'1'的错误数据，并标记为2
    #     # sql_update = "update \"Hosp2_rule_copy\" set \"" + right + "\"='" + result + "' , \"Label\"='2'   where \"Label\"='1' or \"Label\"='2' and \"" + sqlex
    #     # print(sql_update)
    #     # sql_check = "select * from \"" + path + "\"   where  \"" + sqlex#\"" + right + "\"  #(\"Label\"='1' or \"Label\"='2') and
    #     sql_check = "select * from \"" + path + "\"   where  (\"Label\"='1' or \"Label\"='2') and \"" + sqlex  # \"" + right + "\"
    #     # print(sql_check)
    #     cursor.execute(sql_check)
    #     row_check = cursor.fetchall()
    #     row_check = [x[:-1] for x in row_check]
    #     if order == 0:
    #         row_check = [x[::-1] for x in row_check]
    #     # print(row_check)
    #     flag_check=att2label[right]
    #     for row in row_check:
    #         t2 = len(row)
    #         att = list(label2att.values())
    #         # print(right,row[flag_check],result)
    #         if (row[flag_check]!=result):
    #             error_data+=1
    #             for i in range(t2):  # t2
    #                 if i == 0:
    #                     sql_info = "\"" + att[i] + "\"='" + row[i] + "'"
    #                 else:
    #                     sql_info = sql_info + " and \"" + att[i] + "\"='" + row[i] + "'"
    #             # sql_info = sql_info + " and (\"Label\"='1' or \"Label\"='2')"
    #             print("原始：",sql_info)
    #             # row = list(row)
    #             # row[flag_check] = result
    #             # print("row[flag_check]:",row[flag_check])
    #             # print("result",result)
    #             sql_update="update \"" + path + "\" set \"Label\"='2' , \"" + att[flag_check] + "\"='" + result + "' where  " + sql_info + ""
    #             print("Update信息：", sql_update)
    #             cursor.execute(sql_update)
    #             conn.commit()
    
    cursor.close()
    conn.close()
    # if num>200:
    #     break
    
    print("修复完成")
    print("保存修复规则")
    print("规则字典大小", len(rule))
    # print(str(rule))
    f = open('data/save/rules_final.txt', 'w')
    f.write(str(rule))
    f.close()
    with open('data/save/log.txt', 'a') as f:
        f.write("本次共使用规则数量")
        f.write(str(num))
        f.write("规则错误数量")
        f.write(str(error_rule))
        f.write("数据错误数量")
        f.write(str(error_data))
        f.write("__________")
        f.close()


# 这里可用来循环修复直到无新错误数据
# if (iteration_num>0):
#     print(iteration_num)
#     if (error_rule != 0):
#         self.repair(iteration_num-1,path,order)

if __name__ == '__main__':
    # filter("data/hospital/hospital_dirty.csv")
    # repair("data/hospital/hospital_dirty.csv")
    df = pd.read_csv("data/flights/flights_dirty.csv")
    df = df.fillna("null")
    print(df)