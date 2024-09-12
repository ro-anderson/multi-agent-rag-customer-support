# customer_support_chat/app/main.py
from core.settings import get_settings
from core.logger import logger
from services.chat_service import ChatService

def main():
    settings = get_settings()
    logger.info("Starting Customer Support Chat Application")

    chat_service = ChatService(settings)
    chat_service.run()

if __name__ == "__main__":
    main()
