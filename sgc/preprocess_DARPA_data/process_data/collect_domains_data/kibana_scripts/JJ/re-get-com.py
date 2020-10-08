from elasticsearch import Elasticsearch
import csv

# Define config
host = "10.0.0.100"
port = 9200
timeout = 1000
index = "re-cyber-rc-sent*" # Enter GitHub domain here
doc_type = "doc"
size = 10000
body = {
    "_source": {
        #"includes" : ['extension.sentiment_polarity']
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

rewriter = csv.writer(open('re-cyber-comment.csv', 'w'), delimiter=',')

# Process hits here
def process_hits(hits):
    for item in hits:
        #media = None
        #mediaprovider = None
        #stickied = None
        if 'link_id_h' in item['_source']:
            if 'body_m' in item['_source']:
                bodylen = len(item['_source']['body_m'])
            else:
                bodylen = 0
            author = item['_source']['author_h']
            date = item['_source']['created_date']
            polarity = item['_source']['extension']['sentiment_polarity']
            subjectivity = item['_source']['extension']['sentiment_subjectivity']
            id = item['_source']['id_h']
            root = item['_source']['link_id_h']
            parent = item['_source']['parent_id_h']
            subreddit = item['_source']['subreddit']
            subredditid = item['_source']['subreddit_id']
            score = item['_source']['score']
            gilded = item['_source']['gilded']
            controversiality = item['_source']['controversiality']
            '''
            if 'media' in ['_source']:
                media = item['_source']['media']['type']
                mediaprovider = item['_source']['media']['oembed']['provider_name']
            if 'stickied' in ['_source']:
                stickied = item['_source']['stickied']
            '''
            rewriter.writerow([id, author, date, root, parent, str(bodylen), str(polarity), str(subjectivity), str(score), str(gilded), str(controversiality), subreddit, subredditid])


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

# Before scroll, process current batch of hits
process_hits(data['hits']['hits'])

while scroll_size > 0:
    "Scrolling..."
    data = es.scroll(scroll_id=sid, scroll='2m')

    # Process current batch of hits
    process_hits(data['hits']['hits'])

    # Update the scroll ID
    sid = data['_scroll_id']

    # Get the number of results that returned in the last scroll
    scroll_size = len(data['hits']['hits'])
