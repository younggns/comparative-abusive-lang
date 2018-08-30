import os
import argparse
import helper
# For crawling tweets
import tweepy
import time
from datetime import datetime
import pickle
# For text preprocessing
import re
import numpy as np
from nltk.tokenize import TweetTokenizer
from wordsegment import segment, load
# K-fold splits
from sklearn.model_selection import train_test_split, KFold

##########################################################################
############################## Crawl Tweets ##############################
##########################################################################
def getTweetStat(id_str, e_counter_lst, api):
	try:
		tweet_stat = api.get_status(id_str)
	except tweepy.error.RateLimitError:
		print("Rate limit exceeded while processing",tweet_id,"at",str(datetime.now()))
		time.sleep(60 * 15)
		print("Reprocessing from",tweet_id,"at",str(datetime.now()))
		tweet_stat, e_counter_lst = getTweetStat(id_str, e_counter_lst)
	except tweepy.error.TweepError as e:
		tweet_stat = ''
		if e.api_code == 63: # Suspended user account
			e_counter_lst[0] += 1
		elif e.api_code == 144: # Deleted tweet
			e_counter_lst[1] += 1
		elif e.api_code == 179: # Unauthorized to access
			e_counter_lst[2] += 1
		elif e.api_code == 34: # Non-existing tweet
			e_counter_lst[3] += 1
		else:
			pass

	return tweet_stat, e_counter_lst

def getTweetType(status):
	_type = "plain"
	if len(status["entities"]["user_mentions"]) > 0:
		_type = "mention"
	if "retweeted_status" in status:
		_type = "retweet"
	if status["in_reply_to_status_id"] != None:
		_type = "reply"
	if status["is_quote_status"] == True:
		_type = "quote"
	return _type

def crawlTweet():
	######### Modify this part with your Twitter App credentials #########
	consumer_key = ''
	consumer_secret = ''
	access_token = ''
	access_token_secret = ''
	######################################################################

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)

	tweet_id_list = []

	with open('hatespeech_labels.csv') as f:
		lines = f.readlines()[1:]
		for line in lines:
			tweet_id_list.append(line)
	print("Done reading csv data")

	error_counter = [0,0,0,0]

	path = os.path.dirname(os.path.abspath(__file__)) + "/data/crawled"
	if not os.path.exists(path):
		os.makedirs(path)

	ID_Text_file = open(path+'/ID-Text.csv', 'w')
	ID_Type_file = open(path+'/ID-Type.csv', 'w')
	ID_contextID_file = open(path+'/ID-contextID.csv', 'w')

	id_entities = {}

	for id_label_pair in tweet_id_list:
		tweet_id = id_label_pair.split(",")[0]
		label = id_label_pair[len(id_label_pair.split(",")[0])+1:].replace('\n','')

		if (tweet_id_list.index(id_label_pair)+1)%200 == 0:
			print((tweet_id_list.index(id_label_pair)+1),"tweets processed among",len(tweet_id_list))

		tweet_stat, error_counter = getTweetStat(tweet_id, error_counter, api)

		if tweet_stat != '':
			input_tweet_text = tweet_stat._json["text"].replace("\n"," ")
			input_tweet_type = getTweetType(tweet_stat._json)

			ID_Type_file.write(tweet_id+","+input_tweet_type+"\n")
			ID_Text_file.write(tweet_id+","+input_tweet_text+"\n")

			context_tweet_text = ''
			if input_tweet_type == 'reply':
				context_tweet_id = tweet_stat._json["in_reply_to_status_id_str"]
				context_tweet_stat, error_counter = getTweetStat(context_tweet_id, error_counter, api)
				if context_tweet_stat != '':
					context_tweet_text = context_tweet_stat._json["text"].replace("\n"," ")
					context_tweet_type = getTweetType(context_tweet_stat._json)

					ID_Type_file.write(context_tweet_id+","+context_tweet_type+"\n")
					ID_Text_file.write(context_tweet_id+","+context_tweet_text+"\n")
					ID_contextID_file.write(tweet_id+","+context_tweet_id+"\n")
				
			if input_tweet_type == 'quote':
				if "quoted_status_id" not in tweet_stat._json:
					error_counter[-1] += 1
				else:
					context_tweet_id = tweet_stat._json["quoted_status_id_str"]
					context_tweet_stat, error_counter = getTweetStat(context_tweet_id, error_counter, api)
					if context_tweet_stat != '':
						context_tweet_text = context_tweet_stat._json["text"].replace("\n"," ")
						context_tweet_type = getTweetType(context_tweet_stat._json)

						ID_Type_file.write(context_tweet_id+","+context_tweet_type+"\n")
						ID_Text_file.write(context_tweet_id+","+context_tweet_text+"\n")
						ID_contextID_file.write(tweet_id+","+context_tweet_id+"\n")

			id_entities[tweet_id] = [label, input_tweet_text, context_tweet_text]

	print("Done crawling tweets")

	log_output_str = 'Total input tweets: '+str(len(tweet_id_list))+'\n'
	log_output_str += 'Suspended: '+str(error_counter[0])+'\n'
	log_output_str += 'Deleted: '+str(error_counter[1])+'\n'
	log_output_str += 'Unauthorized to access: '+str(error_counter[2])+'\n'
	log_output_str += 'Non-existing: '+str(error_counter[3])

	with open(path+'/log.txt', 'w') as f:
		f.write(log_output_str)

	ID_Text_file.close()
	ID_Type_file.close()
	ID_contextID_file.close()

	print("Done writing raw csv files")

	with open(path+"/data.pkl", "wb") as f:
		pickle.dump(id_entities, f)

	print("Done writing a pickle file")

################################################################################
############################## Text Preprocessing ##############################
################################################################################
def text_preprocess(text, tknzr):
	FLAGS = re.MULTILINE | re.DOTALL
	# Different regex parts for smiley faces
	eyes = r"[8:=;]"
	nose = r"['`\-]?"

	# function so code less repetitive
	def re_sub(pattern, repl):
		return re.sub(pattern, repl, text, flags=FLAGS)

	text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
	text = re_sub(r"/"," / ")
	text = re_sub(r"@\w+", "<user>")
	text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
	text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
	text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
	text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
	text = re_sub(r"<3","<heart>")
	text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
	text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
	text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
	text = re_sub(r"#\S+", lambda hashtag: " ".join(segment(hashtag.group()[1:]))) # segment hastags

	tokens = tknzr.tokenize(text.lower())
	return " ".join(tokens)

def concat_data():
	###################################
	########## Lookup Tables ##########
	###################################
	labels = ["normal", "spam", "hateful", "abusive"]
	num_classes = len(labels)

	label_lookup = np.zeros((num_classes,num_classes),int)
	np.fill_diagonal(label_lookup, 1)
	###################################

	text_data, context_data, label_data = [], [], []
	label_dict = {}
	for i, label in enumerate(labels):
		label_dict[label] = i

	path = os.path.dirname(os.path.abspath(__file__)) + "/data/crawled"
	with open(path+"/data.pkl", "rb") as f:
		id2entities = pickle.load(f)

	load()
	tknzr = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=False)
	for _id in id2entities:
		if id2entities[_id][0] in label_dict.keys():
			text_data.append(text_preprocess(id2entities[_id][1], tknzr))
			context_data.append(text_preprocess(id2entities[_id][2], tknzr))

			label_data.append(label_lookup[ label_dict[id2entities[_id][0]] ])

	assert len(text_data) == len(context_data) == len(label_data)

	return text_data, context_data, label_data

################################################################################
############################## K-fold Data Split ###############################
################################################################################
def kfold_splits(text_data, context_data, label_data, k):
	kfold_text, kfold_context, kfold_label = [], [], []
	for i in range(k):
		_text_data = {"train": {}, "valid": {}, "test": {}}
		_context_data = {"train": {}, "valid": {}, "test": {}}
		_label_data = {"train": {}, "valid": {}, "test": {}}
		kfold_text.append(_text_data)
		kfold_context.append(_context_data)
		kfold_label.append(_label_data)

	kf = KFold(n_splits=k, shuffle=True, random_state=0)
	kfold_index = 0
	for rest_index, test_index in kf.split(text_data):

		train_index, valid_index, _, _ = train_test_split(rest_index, np.zeros_like(rest_index), test_size=0.05)

		kfold_text[kfold_index]["train"] = [text_data[index] for index in train_index]
		kfold_text[kfold_index]["test"] = [text_data[index] for index in test_index]
		kfold_text[kfold_index]["valid"] = [text_data[index] for index in valid_index]

		kfold_context[kfold_index]["train"] = [context_data[index] for index in train_index]
		kfold_context[kfold_index]["test"] = [context_data[index] for index in test_index]
		kfold_context[kfold_index]["valid"] = [context_data[index] for index in valid_index]

		kfold_label[kfold_index]["train"] = [label_data[index] for index in train_index]
		kfold_label[kfold_index]["test"] = [label_data[index] for index in test_index]
		kfold_label[kfold_index]["valid"] = [label_data[index] for index in valid_index]

		assert len(kfold_text[kfold_index]["train"]) == len(kfold_context[kfold_index]["train"]) == len(kfold_label[kfold_index]["train"])
		assert len(kfold_text[kfold_index]["valid"]) == len(kfold_context[kfold_index]["valid"]) == len(kfold_label[kfold_index]["valid"])
		assert len(kfold_text[kfold_index]["test"]) == len(kfold_context[kfold_index]["test"]) == len(kfold_label[kfold_index]["test"])

		train_length = len(kfold_text[kfold_index]["train"])
		valid_length = len(kfold_text[kfold_index]["valid"])
		test_length = len(kfold_text[kfold_index]["test"])

		kfold_index += 1

	print("Input Data Splitted: %s (train) / %s (valid) / %s (test)" % (train_length, valid_length, test_length))
	
	return kfold_text, kfold_context, kfold_label

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--download_source", type=helper.str2bool, default='True')
	parser.add_argument("--crawl_tweets", type=helper.str2bool, default='True')
	parser.add_argument("--feature_level", type=helper.str2feature, default='word')

	args = vars(parser.parse_args())

	if args["download_source"] == True:
		print("\nDownloading the source file from the github repository...")
		os.system("curl -LJO https://raw.githubusercontent.com/ENCASEH2020/hatespeech-twitter/master/hatespeech_labels.csv")
		print("Downloaded.\n")

	if args["crawl_tweets"] == True:
		crawlTweet()

	_text, _ctxt, _label = concat_data()
	_text_split, _ctxt_split, _label_split = kfold_splits(_text, _ctxt, _label, 10)

	path = os.path.dirname(os.path.abspath(__file__)) + "/data/preprocessed/" + args["feature_level"]
	if not os.path.exists(path):
		os.makedirs(path)
	with open(path+"/data_splits.pkl", "wb") as f:
		print("Creating splits pickle files to",path)
		pickle.dump({"text_data": _text_split, "context_data": _ctxt_split, "label_data": _label_split}, f)
