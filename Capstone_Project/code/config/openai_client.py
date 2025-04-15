# Capstone_Project/code/config/openai_client.py

from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()  # .env 파일에서 환경변수 로드

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY not found in environment. Please check your .env file.")

client = OpenAI(api_key=api_key)
