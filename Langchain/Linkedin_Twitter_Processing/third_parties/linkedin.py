import os
import requests


def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}

    #  if u want to scrape your own LinkedIn uncomment this
    # response = requests.get(
    #     api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    # )
    # return response

    gist_response = requests.get(
        "https://gist.githubusercontent.com/MatrixX-X/bc80cab19e73d1b5a0d523701116fb32/raw/ea43e7f65c13c9fb3e3164e3b3d282fb044ee048/eden-marco.json"
    )
    # return gist_response

    # Cleans the unnecessary dat that we are receiving from this function

    data = gist_response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
