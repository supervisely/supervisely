class DialogWindowError(Exception):
    def __init__(self, title, description):
        self.title = title
        self.description = description
        super().__init__(self.get_message())

    def get_message(self):
        return f"{self.title}: {self.description}"

    def __str__(self):
        return self.get_message()
