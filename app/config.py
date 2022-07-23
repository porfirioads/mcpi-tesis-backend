from pydantic import BaseSettings
import os
import logging


class Settings(BaseSettings):
    class Config:
        APP_NAME: str = os.getenv('APP_NAME', 'Example')
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()
logger = logging.getLogger('uvicorn')
logger.setLevel(logging.DEBUG)
