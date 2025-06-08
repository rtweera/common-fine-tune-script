import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_URL = os.getenv("API_URL")
API_TOKEN = os.getenv("API_KEY")
CORRELATION_ID = os.getenv("CORR_ID")

HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Authorization": f"Bearer {API_TOKEN}",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://console.choreo.dev",
    "Referer": "https://console.choreo.dev/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "correlation-id": f"{CORRELATION_ID}",
}
