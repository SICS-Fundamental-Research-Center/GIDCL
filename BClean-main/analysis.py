def analysis(actual_error, repair_error,dirty_data,clean_data):
	actual_error_repiair = {}
	whole_error = len(actual_error)
	whole_repair = len(repair_error)
	print("+++all repair:{}".format(whole_repair))
	pre_not_right = {}
	miss_err = 0
	pre_right = 0
	pre_wrong = 0
	missing_wrong = {}
	
	for cell in actual_error:
		if (cell in repair_error):
			if (actual_error[cell] == repair_error[cell]):
				pre_right += 1
				actual_error_repiair[cell] = {actual_error[cell]: "{} ===> {}".format(
					dirty_data.loc[cell[0], cell[1]], repair_error[cell])}
		else:
			missing_wrong[cell] = "{} ===> {}".format(
				dirty_data.loc[cell[0], cell[1]], clean_data.loc[cell[0], cell[1]])
			miss_err += 1
	for cell in repair_error:
		if (cell not in actual_error):
			pre_wrong += 1
			pre_not_right[cell] = "{} ===> {} but not {}".format(
				dirty_data.loc[cell[0], cell[1]], clean_data.loc[cell[0], cell[1]], repair_error[cell])
		else:
			if (actual_error[cell] != repair_error[cell]):
				pre_wrong += 1
				pre_not_right[cell] = [repair_error[cell]]
	
	P = pre_right / whole_repair
	R = pre_right / (whole_error)
	F1 = 2 * (P * R) / (P + R)
	
	print("lack of ")
	print("miss_err:{}, pre_right:{}".format(miss_err, pre_right))
	
	return P, R, F1