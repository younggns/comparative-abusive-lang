import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import helper	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--feature_level", type=helper.str2feature, default='word')
	parser.add_argument("--clf", type=helper.str2clfs, required=True)

	args = vars(parser.parse_args())

	if args["clf"] in ["NB", "LR", "SVM", "RF", "GBT"]:
		os.system("python3 wrapper_ml.py --feature_level "+args["feature_level"]+" --clf "+args["clf"])
	else:
		os.system("python3 wrapper_nn.py --feature_level "+args["feature_level"]+" --clf "+args["clf"])