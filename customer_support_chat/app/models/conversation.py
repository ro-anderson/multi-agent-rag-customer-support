# customer_support_chat/app/models/conversation.py

class Conversation:
    def __init__(self, user_id):
        self.user_id = user_id
        self.history = []

    def add_message(self, message):
        self.history.append(message)
