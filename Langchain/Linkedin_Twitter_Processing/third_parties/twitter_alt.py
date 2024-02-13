import pandas as pd
from ntscraper import Nitter


def scrape_user_tweets(username, num=5):
    """
    Scrapes a Twitter user's original tweets (i.e., not retweets or replies) and returns them as a list of dictionaries.
    Each dictionary has three fields: "time_posted" (relative to now), "text", and "url".
    """
    scraper = Nitter(0)
    tweets = scraper.get_tweets(username, mode="user", number=num)

    final_tweets = []
    for x in tweets["tweets"]:
        data = [x["link"], x["text"]]
        final_tweets.append(data)

    # Set display options to show all text in DataFrame
    pd.set_option("display.max_colwidth", None)

    data = pd.DataFrame(final_tweets, columns=["twitter_link", "text"])

    tweet_list = []
    for index, row in data.iterrows():
        if "elonmusk" in row["twitter_link"]:
            tweet_dict = {}
            tweet_dict["text"] = row["text"]
            tweet_dict["url"] = row["twitter_link"]
            tweet_list.append(tweet_dict)

    return tweet_list


if __name__ == "__main__":
    print(scrape_user_tweets(username="elonmusk", num=5))
