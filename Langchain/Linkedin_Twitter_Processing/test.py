from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Now you can access the environment variables using os.environ
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
