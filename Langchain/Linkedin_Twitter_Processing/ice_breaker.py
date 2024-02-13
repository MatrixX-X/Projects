from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_agent_lookup import lookup as linkedin_agent_lookup

from third_parties.twitter import scrape_user_tweets

from agents.twitter_agent_lookup import lookup as twitter_agent_lookup

information = """
Abdul Kalam Ghulam Muhiyuddin Ahmed bin Khairuddin Al-Hussaini Azad ((listenⓘ); 11 November 1888 – 22 February 1958) was an Indian independence activist, writer and a senior leader of the Indian National Congress. Following India's independence, he became the First Minister of Education in the Indian government. He is commonly remembered as Maulana Azad; the word Maulana is an honorific meaning 'Our Master' and he had adopted Azad (Free) as his pen name. His contribution to establishing the education foundation in India is recognised by celebrating his birthday as National Education Day across India.[2][3]

As a young man, Azad composed poetry in Urdu, as well as treatises on religion and philosophy. He rose to prominence through his work as a journalist, publishing works critical of the British Raj and espousing the causes of Indian nationalism. Azad became the leader of the Khilafat Movement, during which he came into close contact with the Indian leader Mahatma Gandhi. After the failure of the Khilafat Movement, he became closer to the Congress.[4] Azad became an enthusiastic supporter of Gandhi's ideas of non-violent civil disobedience, and worked to organise the non-co-operation movement in protest of the 1919 Rowlatt Acts. Azad committed himself to Gandhi's ideals, including promoting Swadeshi (indigenous) products and the cause of Swaraj (Self-rule) for India. In 1923, at an age of 35, he became the youngest person to serve as the President of the Indian National Congress.
"""
name = "Eden Marco"
if __name__ == "__main__":
    print("Hello LangChain")

    linkedin_profile_url = linkedin_agent_lookup(name=name)  # name = name
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    twitter_username = twitter_agent_lookup(name=name)
    tweets = scrape_user_tweets(username="twitter_username", num_tweets=2)

    summary_template = """
         given the Linkedin information {linkedin_information} and twitter {twitter_information} about a person from I want you to create:
         1. a short summary
         2. two interesting facts about them
         3. A topic that may interest them
         4. 2 creative Ice breakers to open a conversation with them 
     """

    prompt = PromptTemplate(
        input_variables=["linkedin_information", "twitter_information"],
        template=summary_template,
    )

    llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=prompt)

    # returns whole result
    print(chain.invoke(input={"linkedin_information": linkedin_data, "twitter_information": tweets}))

    # # Pass the required input to the invoke method
    # result = chain.invoke({"information": linkedin_data})
    #
    # # Access and print the 'text' part from the result
    # print(result["text"])

