import argparse
import re

def str2bool(v):
	if v == 'True':
		return True
	elif v == 'False':
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def str2feature(v):
	if v in ['word', 'Word', 'WORD']:
		return 'word'
	elif v in ['char', 'Char', 'CHAR', 'character', 'Character', 'CHARACTER']:
		return 'char'
	elif v in ['hybrid', 'Hybrid', 'HYBRID']:
		return 'hybrid'
	else:
		raise argparse.ArgumentTypeError("'word' or 'char' expected.")

def str2ngrams(v):
	ngram_list = v.split(',')
	if len(ngram_list) == 2:
		b_i = int(re.sub("\D", "", ngram_list[0]))
		e_i = int(re.sub("\D", "", ngram_list[1]))
		if e_i < b_i:
			raise argparse.ArgumentTypeError("Invalid range.")
		else:
			return (b_i, e_i)
	else:
		raise argparse.ArgumentTypeError("n-gram range is expected. e.g. '1,3'")

def str2clfs(v):
	if v in ["nb", "NB", "Nb"]:
		return 'NB'
	elif v in ["lr", "LR", "Lr"]:
		return 'LR'
	elif v in ["svm", "SVM", "Svm"]:
		return 'SVM'
	elif v in ["rf", "RF", "Rf"]:
		return 'RF'
	elif v in ["gbt", "GBT", "Gbt"]:
		return 'GBT'
	elif v in ["cnn", "CNN", "Cnn"]:
		return "CNN"
	elif v in ["rnn", "RNN", "Rnn"]:
		return "RNN"
	else:
		raise argparse.ArgumentTypeError("Invalid classifier name.")


