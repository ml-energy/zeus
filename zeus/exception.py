class ZeusBaseException(Exception):
    """Zeus base exception class.""" 

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message