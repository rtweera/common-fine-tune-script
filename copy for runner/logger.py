import os 
from dotenv import load_dotenv
import requests
import json

class Logger:
    def __init__(self):
        load_dotenv()
        self.endpoint = os.getenv("BETTERSTACK_URL")
        self.bearer_token = os.getenv("BETTERSTACK_BEARER_TOKEN")
        if not self.endpoint or not self.bearer_token:
            raise ValueError("BETTERSTACK_URL and BETTERSTACK_BEARER_TOKEN must be set in the environment variables.")
        self.headers = {
            "Authorization": self.bearer_token,
            "Content-Type": "application/json"
        }

    def log(self, message, level="info"):
        try:
            payload = {
                "dt": r"$(date -u +'%Y-%m-%d %T UTC')",
                "message": f"[{level}] {message}",
            }
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                verify=True  # Disable SSL verification (this is in the curl they have given)
            )
            if response.status_code == 202:
                return True
            else:
                print(f"Failed to send log. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"Error sending log: {str(e)}")
            return False
    def info(self, message):
        """Log an INFO level message"""
        return self.log(message, "INFO")
    
    def error(self, message):
        """Log an ERROR level message"""
        return self.log(message, "ERROR")
    
    def warning(self, message):
        """Log a WARNING level message"""
        return self.log(message, "WARNING")
    
    def debug(self, message):
        """Log a DEBUG level message"""
        return self.log(message, "DEBUG")