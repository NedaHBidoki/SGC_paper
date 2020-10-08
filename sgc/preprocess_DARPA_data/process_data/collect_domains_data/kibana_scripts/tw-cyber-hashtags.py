{
  "aggs": {
    "2": {
      "terms": {
        "field": "id_h.keyword",
        "size": 1000,
        "order": {
          "_count": "desc"
        }
      },
      "aggs": {
        "3": {
          "terms": {
            "field": "hashtags.keyword",
            "size": 100,
            "order": {
              "_count": "desc"
            }
          }
        }
      }
    }
  },
  "size": 0,
  "_source": {
    "excludes": []
  },
  "stored_fields": [
    "*"
  ],
  "script_fields": {},
  "docvalue_fields": [
    {
      "field": "created_at",
      "format": "date_time"
    },
    {
      "field": "quoted_status.created_at",
      "format": "date_time"
    },
    {
      "field": "quoted_status.user.created_at",
      "format": "date_time"
    },
    {
      "field": "retweeted_status.created_at",
      "format": "date_time"
    },
    {
      "field": "retweeted_status.quoted_status.created_at",
      "format": "date_time"
    },
    {
      "field": "retweeted_status.quoted_status.user.created_at",
      "format": "date_time"
    },
    {
      "field": "retweeted_status.user.created_at",
      "format": "date_time"
    },
    {
      "field": "timestamp_ms",
      "format": "date_time"
    },
    {
      "field": "user.created_at",
      "format": "date_time"
    }
  ],
  "query": {
    "bool": {
      "must": [
        {
          "match_all": {}
        },
        {
          "match_all": {}
        },
        {
          "range": {
            "created_at": {
              "gte": 1420070400000,
              "lte": 1522540800000,
              "format": "epoch_millis"
            }
          }
        }
      ],
      "filter": [],
      "should": [],
      "must_not": []
    }
  }
}
