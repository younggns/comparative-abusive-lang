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
	else:
		raise argparse.ArgumentTypeError("'word' or 'char' expected.")

def str2embedding(v):
	if v in ['no', 'No', 'NO']:
		return 'no'
	elif v in ['yes', 'Yes', 'YES']:
		return 'yes'
	else:
		raise argparse.ArgumentTypeError("'yes' or 'no' expected.")

def str2ngrams(v):
	ngram_list = v.split(',')
	if len(v) == 2:
		b_i = int(re.sub("\D", "", ngram_list[0]))
		e_i = int(re.sub("\D", "", ngram_list[1]))
		if e_i < b_i:
			raise argparse.ArgumentTypeError("Invalid range.")
		else:
			return (b_i, e_i)
	else:
		raise argparse.ArgumentTypeError("n-gram range is expected. e.g. '1,3'")