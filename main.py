import re
import os
import json
import spacy
import tweepy
import pickle
import gensim
import string
import zipfile
import argparse
import numpy as np
import pandas as pd
import urllib.request
from itertools import chain
from sklearn.svm import SVC
from collections import Counter
import matplotlib.pyplot as plt
from spacy.lemmatizer import Lemmatizer
from sklearn.preprocessing import LabelEncoder
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='Select train or stream mode.')
args = parser.parse_args()
mode = args.mode

negative_tweets = []
nlp = spacy.load('en_core_web_sm')
vector_len = 300
w2v_epochs = 150
samples = 10000
keyword = 'trump'

class MyStreamListener(tweepy.streaming.StreamListener):
	def on_status(self, status):

		tweet = cleansing(status.text.lower())
		vector = word_vector(tweet, vector_len, w2v_model)
		y_pred = model.predict(vector)

		decode_map = {0: "negative", 1: "positive"}
		results[decode_map[int(y_pred)]] += 1

		if decode_map[int(y_pred)] == "negative":

			negative_tweets.append(tweet)
			counter_obj = Counter(chain.from_iterable(negative_tweets))
			most_common = counter_obj.most_common(10)
			if len(most_common) < 10:
				return
			counts = [most_common[0][1], most_common[1][1], most_common[2][1], most_common[3][1], \
					most_common[4][1], most_common[5][1], most_common[6][1], most_common[7][1], \
					most_common[8][1], most_common[9][1]]
			values = [most_common[0][0], most_common[1][0], most_common[2][0],  most_common[3][0], \
					most_common[4][0], most_common[5][0], most_common[6][0], most_common[7][0], \
					most_common[8][0], most_common[9][0]]
			plt.title('Most frequent negative-sentiment words.')
			plt.bar(values, counts)
			plt.draw()
			plt.pause(0.25)
			plt.clf()

def word_vector(tokens, size, w2v_model):

	vec = np.zeros(size).reshape((1, size))
	count = 0.
	for word in tokens:
		try:
			vec += w2v_model.wv[word].reshape((1, size))
			count += 1.
		except KeyError:
			continue
	if count != 0:
		vec /= count
	return vec

def word2vec(text):

	w2v_model = gensim.models.word2vec.Word2Vec(text, size = vector_len, window = 5, min_count = 5, workers = 4)
	w2v_model.train(text, total_examples = len(text), epochs = w2v_epochs)
	wordvec_arrays = np.zeros((len(text), vector_len))
	for i in range(len(text)):
		wordvec_arrays[i,:] = word_vector(text.values[i], vector_len, w2v_model)
	wordvec_df = pd.DataFrame(wordvec_arrays)
	return wordvec_df, w2v_model

def cleansing(text):

	emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
		  ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
		  ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
		  ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
		  '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
		  '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
		  ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

	urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
	userPattern       = '@[^\s]+'
	alphaPattern      = "[^a-zA-Z0-9]"
	sequencePattern   = r"(.)\1\1+"
	seqReplacePattern = r"\1\1"

	text = re.sub(urlPattern,' ',text)
	text = re.sub(userPattern,' ', text)
	text = re.sub(alphaPattern, ' ', text)
	text = re.sub(sequencePattern, seqReplacePattern, text)

	for emoji in emojis:
		text = text.replace(emoji, "EMOJI" + emojis[emoji])

	text  = "".join([char for char in text if char not in string.punctuation]) # punctuation
	text = re.sub('[0-9]+', '', text) # punctuation
	text = [token.lemma_ for token in nlp(text)] # tokenize + lemmatize
	text = [word for word in text if word not in STOP_WORDS] # remove stopwords
	text = [word.strip(' ') for word in text] # remove spaces
	text = list(filter(None, text)) # remove empry str
	text = [word for word in text if len(word) > 2 and word != "-PRON-"] # remove single chars

	return text

if mode == 'train':

	# dataset
	if not (os.path.isfile('data/dataset.zip')):
		print('Downloading dataset...')
		with urllib.request.urlopen('http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip') as dataset:
			with open('data/dataset.zip', 'wb') as out_file:
				out_file.write(dataset.read())

	columns = ["target", "ids", "date", "flag", "user", "text"]
	encoding = 'ISO-8859-1'
	dataset_zip = zipfile.ZipFile('data/dataset.zip') 
	dataset_full = pd.read_csv(dataset_zip.open('training.1600000.processed.noemoticon.csv'), encoding=encoding, names=columns, usecols=['target', 'text'])

	dataset_full = dataset_full.sample(frac=1).reset_index(drop=True)
	dataset = dataset_full.copy().sample(samples)
	dataset = dataset[['target', 'text']]
	dataset.dropna(inplace=True)

	print('Dataset cleansing...')
	dataset['text'] = dataset['text'].apply(lambda x: cleansing(x.lower()))

	print('Starting word2vec...')
	wordvec_df, w2v_model = word2vec(dataset['text'])

	print('Dataset shape after word2vec: ', wordvec_df.shape)
	decode_map = {0: "negative",  4: "positive"}
	dataset.target = dataset.target.apply(lambda x: decode_map[int(x)])
	encoder = LabelEncoder()
	target = encoder.fit_transform(dataset['target'])

	print(dataset.target.value_counts())
	X_train, X_test, y_train, y_test = train_test_split(wordvec_df, target, test_size=0.2)
	print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

	tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['scale'],'C': [1]}]
	print('Starting SVM...')
	model = GridSearchCV(SVC(), tuned_parameters, scoring='f1', n_jobs=4, cv=5)
	model.fit(X_train, y_train)

	print("Best parameters set found on development set:")
	print(model.best_params_)

	# save both models
	pickle.dump(model, open('models/model.sav', 'wb'))
	w2v_model.save('models/word2vec.sav')

	y_pred = model.predict(X_test)
	print(classification_report(y_test, y_pred))

if mode == 'stream':

	results = {'positive': 0, 'negative': 0}
	
	credentials_file = open('credentials.json')
	credentials = json.load(credentials_file)

	ACCESS_TOKEN = credentials['ACCESS_TOKEN']
	ACCESS_TOKEN_SECRET = credentials['ACCESS_TOKEN_SECRET']
	CONSUMER_KEY = credentials['CONSUMER_KEY']
	CONSUMER_SECRET = credentials['CONSUMER_SECRET']

	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
	api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

	# load both models
	model = pickle.load(open('models/model.sav', 'rb'))
	w2v_model = gensim.models.Word2Vec.load("models/word2vec.sav")

	stream = tweepy.Stream(auth=api.auth, listener=MyStreamListener())
	stream.filter(track=[keyword])

	try:
		print('Start streaming.')
		stream.sample(languages=['en'])
	except KeyboardInterrupt:
		print("Stopped.")
	finally:
		print('Done.')
		stream.disconnect()
