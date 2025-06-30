from crewai_tools import SerperDevTool
import os
import sys

API_KEY = os.getenv('SERPER_API_KEY')
print("SERPER_API_KEY =", API_KEY, file=sys.stderr)

if API_KEY is None:
    raise ValueError("SERPER_API_KEY not found in environment variables.")

google_search_tool = SerperDevTool(
    api_key=API_KEY
)
