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


file_ = '/home/social-sim/Documents/SocialSimCodeTesting/TH/TH-analysis/Data/user_variables/sample.csv'
user_attr = pd.read_csv(file_)
user_attr = user_attr.dropna()
texts = user_attr["user.description_m.keyword: Descending"]
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



import gensim
def topic_modeling():
	NUM_TOPICS = 3
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15, minimum_probability=0)
	ldamodel.save('model5.gensim')
	topics = ldamodel.print_topics(num_words=4)
	#for topic in topics:
	#   print(topic)
	print(ldamodel.print_topic(1))



import pyLDAvis.gensim
def visualize():
	lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
	pyLDAvis.display(lda_display)
	pyLDAvis.save_html(lda_display,'topics.html')




dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
topics = lda.print_topics(num_words=4)
#for topic in topics:
#   print(topic)

#num_topics = 3
#topic_words = []
#for i in range(num_topics):
#	tt = lda.get_topic_terms(i,20)
#	topic_words.append([dictionary[pair[0]] for pair in tt])
#	print(topic_words)
lda_corpus = lda[corpus]
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)


usres_topic=[]
for id_,corp in enumerate(lda_corpus):
	topics = []
	for t_id,t in enumerate(corp[0:]):
		if corp[t_id][1]>threshold:
			topics.append(t_id)
	usres_topic.append(topics)
user_attr['topics']=usres_topic

user_attr.to_csv('user_attr_matrix.csv')

print(user_attr['topics'])
