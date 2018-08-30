import argparse

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