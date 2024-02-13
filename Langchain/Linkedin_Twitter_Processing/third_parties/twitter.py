import os
import logging

import tweepy

logger = logging.getLogger("twitter")

twitter_client = tweepy.Client(
    bearer_token=os.environ["TWITTER_BEARER_TOKEN"],
    consumer_key=os.environ["TWITTER_API_KEY"],
    consumer_secret=os.environ["TWITTER_API_KEY_SECRET"],
    access_token=os.environ["TWITTER_ACCESS_TOKEN"],
    access_token_secret=os.environ["TWITTER_ACCESS_TOKEN_SECRET"],
)


def scrape_user_tweets(username, num_tweets=5):
    """
    Scrapes a Twitter user's original tweets (i.e., not retweets or replies) and returns them as a list of dictionaries.
    Each dictionary has three fields: "time_posted" (relative to now), "text", and "url".
    """

    # uncomment when using twitter API
    # user_id = twitter_client.get_user(username=username).data.id
    # tweets = twitter_client.get_users_tweets(
    #     id=user_id, max_results=num_tweets, exclude=["retweets", "replies"]
    # )

    data = [
        {
            "edit_history_tweet_ids": ["1659558201040203777"],
            "id": "1659558201040203777",
            "text": "Here's some code \ud83d\udc47 and the Github repo link -\nhttps://t.co/jkyvEX5WiG\nWhere you can find a reference for what we have discussed.\nInspired by @hwchase17 \ud83d\udc51\ud83d\udc51 the creator of @LangChainAI \n\nEND - OF -THREAD",
        },
        {
            "edit_history_tweet_ids": ["1659558198649470977"],
            "id": "1659558198649470977",
            "text": "@LangChainAI  has tons of more stuff on it, making our lives so much easier when developing LLM powered applications and IMO is the go to open source framework for developing LLM applications. \nhttps://t.co/Il4HMgDVcI",
        },
        {
            "edit_history_tweet_ids": ["1659558196577443840"],
            "id": "1659558196577443840",
            "text": "8/8 And here's where @LangChainAI  \ud83e\udd9c\ud83d\udd17shines - it automates this entire process and does all the heavy lifting for us \ud83d\udcaa\nWith one line of code, you can invoke a RetrievalQA chain that takes care of all those tasks. https://t.co/Q5XDNtD9Ce",
        },
        {
            "edit_history_tweet_ids": ["1659558193217896452"],
            "id": "1659558193217896452",
            "text": "7/8 Now we  need convert these vectors back into readable text \ud83d\udcdd and send it as context to the LLM with our original query. This augmented prompt helps the LLM provide an accurate answer, even though it wasn't originally trained on our specific library \ud83c\udfcb\ufe0f\u200d\u2640\ufe0f. https://t.co/bHQ4lO7iIu",
        },
        {
            "edit_history_tweet_ids": ["1659558190843846657"],
            "id": "1659558190843846657",
            "text": "6/8: Now we retrieve these vectors. They contain the context required to answer our query \ud83d\udd75\ufe0f\u200d\u2642\ufe0f and represent the text that we need to base our answer on. https://t.co/180EXiQbAr",
        },
    ]

    tweet_list = []
    for tweet in data:  # tweets.data if using API
        tweet_dict = {}
        tweet_dict["text"] = tweet["text"]
        tweet_dict["url"] = f"https://twitter.com/{username}/status/{tweet['id']}"
        tweet_list.append(tweet_dict)

    return tweet_list


if __name__ == "__main__":
    print(scrape_user_tweets(username="hwchase17"))
