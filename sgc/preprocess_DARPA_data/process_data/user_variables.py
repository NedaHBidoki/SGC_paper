import pandas as pd
import sys 
import random
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser = English()
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
from nltk.corpus import wordnet as wn
from gensim import corpora
import pickle
from itertools import chain

db = 'twitter_cve'
base = '/home/social-sim/Documents/SocialSimCodeTesting/gcn/preprocess_DARPA_data/raw_data/%s/'%db
file_ = base + '/attr_74074.csv'
user_attr = pd.read_csv(file_, thousands=',')#, nrows=1000)
user_attr = user_attr.dropna()
cols = ['id_h']
user_attr = user_attr.set_index('id_h')
hash_df = user_attr.groupby(user_attr.index)["hashtags.keyword: Descending"].apply(lambda x: "{%s}" % ', '.join(x))
user_attr = user_attr[["extension.polarity: Descending","extension.subjectivity: Descending","possibly_sensitive: Descending","user.followers_count: Descending","user.friends_count: Descending"]].groupby(cols).max() ### max or min or mean??
#print(hash_df.index)
#print('result:',len(result))
texts = hash_df#user_attr["hashtags.keyword: Descending"]
#print(texts.head())

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def prepare_text_for_lda(text):
	try:
		tokens = tokenize(text)
		tokens = [token for token in tokens if len(token) > 4]
		tokens = [token for token in tokens if token not in en_stop]
		tokens = [get_lemma(token) for token in tokens]
		return tokens
	except:
		print(pd.isnull(text))
		sys.exit()

def get_dict():
	text_data = []
	for line in texts.tolist():
		if not pd.isnull(line):
			tokens = prepare_text_for_lda(line)
			#if random.random() > .99:
			#print(tokens)
			text_data.append(tokens)
		else:
			continue
	dictionary = corpora.Dictionary(text_data)
	corpus = [dictionary.doc2bow(text) for text in text_data]
	pickle.dump(corpus, open('corpus.pkl', 'wb'))
	dictionary.save('dictionary.gensim')
	return dictionary, corpus



import gensim
def topic_modeling():
	NUM_TOPICS = 7
	dictionary, corpus = get_dict()
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15, minimum_probability=0)
	ldamodel.save('model5.gensim')
	topics = ldamodel.print_topics(num_words=4)
	#for topic in topics:
	#   print(topic)
	#print(ldamodel.print_topic(1))



import pyLDAvis.gensim
def visualize():
	lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
	pyLDAvis.display(lda_display)
	pyLDAvis.save_html(lda_display,'topics.html')


topic_modeling()
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
topics = lda.print_topics(num_words=4)
#for topic in topics:
#   print(topic)

num_topics = 5
topic_words = []
print('Users Hashtags:',texts[0:5])
print('#############################')
print("All topics:")
for i in range(num_topics):
	tt = lda.get_topic_terms(i,20)
	topic_words.append([dictionary[pair[0]] for pair in tt])
	print(topic_words)
lda_corpus = lda[corpus]
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
usres_topic=list()
for id_,corp in enumerate(lda_corpus):
	topics = []
	for t_id,t in enumerate(corp[0:]):
		if corp[t_id][1]>threshold:
			topics.append(t_id)
	usres_topic.append(topics)

#print(usres_topic)
#print('len topics:',len(usres_topic))
#print('len rest_df:',len(user_attr))

user_attr['topics'] = pd.Series(usres_topic, index=user_attr.index)
test = user_attr[user_attr['topics'].apply(lambda x:len(x)>2)]
#print('len test:',len(test))
'''
user_attr = pd.DataFrame(user_attr.topics.dropna().tolist(),index=user_attr.index).stack()
user_attr = user_attr.reset_index([0, 'id_h'])
user_attr = user_attr.rename(columns={0: "topics"})
user_attr = user_attr.dropna()
user_attr['topics'] = [str(x) for x in user_attr['topics']]
user_attr = user_attr[user_attr['topics']!= '']
user_attr['topics'] = 'topic_' + user_attr['topics'].astype(str)
'''
import numpy as np
lst_col = 'topics'
user_attr = user_attr.reset_index()
user_attr = pd.DataFrame({col:np.repeat(user_attr[col].values, user_attr[lst_col].str.len())for col in user_attr.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(user_attr[lst_col].values)})[user_attr.columns.tolist()]
#print('user_attr.head:',user_attr.head())
#print('user_attr.columns:',user_attr.columns)
#print('len rest_df:',len(user_attr))
user_attr = user_attr.set_index('id_h')
print(user_attr.columns)
user_attr.to_csv(base + '%s_max_topic_attr.csv'%db, index=True)
print(user_attr.index)
sys.exit()
#result = pd.merge(hash_df, rest_df, left_index=True, right_index=True)
#print('rest_df.columns:',rest_df.columns)
#print('user_attr.head:',user_attr.head())
#print('user_attr.columns:',user_attr.columns)
#user_attr.to_csv('user_attr_matrix.csv')

#print(user_attr['topics'])
