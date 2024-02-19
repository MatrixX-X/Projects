from typing import Tuple
from agents.linkedin_agent_lookup import lookup as linkedin_agent_lookup
from agents.twitter_agent_lookup import lookup as twitter_agent_lookup
from chains.custom_chains import (
    get_summary_chain,
    get_interests_chain,
    get_ice_breaker_chain,
)
from third_parties.linkedin import scrape_linkedin_profile

from third_parties.twitter import scrape_user_tweets
from output_parsers import (
    summary_parser,
    topics_of_interest_parser,
    ice_breaker_parser,
    Summary,
    IceBreaker,
    TopicOfInterest,
)


def ice_break(name: str) -> Tuple[Summary, IceBreaker, TopicOfInterest, str]:
    linkedin_username = linkedin_agent_lookup(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    twitter_username = twitter_agent_lookup(name=name)
    tweets = scrape_user_tweets(username=twitter_username)

    summary_chain = get_summary_chain()
    summary_and_facts = summary_chain.run(
        information=linkedin_data, twitter_posts=tweets
    )
    summary_and_facts = summary_parser.parse(summary_and_facts)

    interests_chain = get_interests_chain()
    interests = interests_chain.run(information=linkedin_data, twitter_posts=tweets)
    interests = topics_of_interest_parser.parse(interests)

    ice_breaker_chain = get_ice_breaker_chain()
    ice_breakers = ice_breaker_chain.run(
        information=linkedin_data, twitter_posts=tweets
    )
    ice_breakers = ice_breaker_parser.parse(ice_breakers)

    return (
        summary_and_facts,
        interests,
        ice_breakers,
        linkedin_data.get("profile_pic_url"),
    )

    # # Pass the required input to the invoke method
    # result = chain.invoke({"information": linkedin_data})
    #
    # Access and print the 'text' part from the result


if __name__ == "__main__":
    pass
