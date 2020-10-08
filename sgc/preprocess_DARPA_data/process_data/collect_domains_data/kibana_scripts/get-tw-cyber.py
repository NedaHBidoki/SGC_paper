from elasticsearch import Elasticsearch
import csv
import sys
import os
# Define config
host = "10.0.0.100"
port = 9200
timeout = 1000
index = "tw-cyber*-tweets" # Enter GitHub domain here
doc_type = "doc"
size = 100
log_file = 'tw-cyber*-tweets.csv'
if os.path.exists(log_file):
		os.remove(log_file)
body = {
    "_source": {
        "includes" : ["user", "hashtags", "possibly_sensitive"]
    },
    "query": {
        "bool": {
        "must": [
            {
            "match_all": {}
            },
        
        ]
        }
    }
}

user_list = []

# Init Elasticsearch instance
es = Elasticsearch(
    [
        {
            'host': host,
            'port': port
        }
    ],
    timeout=timeout
)

rewriter = csv.writer(open(log_file, 'w'), delimiter=',')
rewriter.writerow(['id_h', 'hashtags', 'possibly_sensitive', 'user_time_zone', 'friends_count', 'followers_count'])
# Process hits here
def process_hits(hits):
	for item in hits:
		#sys.exit()
		#media = None
		#mediaprovider = None
		#stickied = None
		#if 'id_h.keyword' in item['_source']:
		id_h = item['_source']['user']['id_h']
		hashtags = item['_source']["hashtags"]
		if 'possibly_sensitive' in item['_source']:
			possibly_sensitive = item['_source']['possibly_sensitive']
		else:
			possibly_sensitive = None
		user_time_zone = item['_source']['user']['time_zone']
		friends_count = item['_source']['user']['friends_count']
		followers_count = item['_source']['user']['followers_count']
		print('writing...')
		rewriter.writerow([id_h, hashtags, possibly_sensitive, user_time_zone, friends_count, followers_count])


# Check index exists
if not es.indices.exists(index=index):
    print("Index " + index + " not exists")
    exit()

# Init scroll by search
data = es.search(
    index=index,
    doc_type=doc_type,
    scroll='1m',
    size=size,
    body=body
)

# Get the scroll ID
sid = data['_scroll_id']
scroll_size = len(data['hits']['hits'])
print(scroll_size)
# Before scroll, process current batch of hits
#process_hits(data['hits']['hits'])

while scroll_size > 0:
    print("Scrolling...")
    data = es.scroll(scroll_id=sid, scroll='2m')

    # Process current batch of hits
    process_hits(data['hits']['hits'])

    # Update the scroll ID
    sid = data['_scroll_id']

    # Get the number of results that returned in the last scroll
    scroll_size = len(data['hits']['hits'])
