from zeus.exception import ZeusBaseError


class ZeusBSOServerBaseError(ZeusBaseError):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.status_code = 500


class ZeusBSOJobSpecMismatchError(ZeusBSOServerBaseError):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.status_code = 409


class ZeusBSOValueError(ZeusBSOServerBaseError):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.status_code = 400


class ZeusBSOOperationOrderError(ZeusBSOServerBaseError):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.status_code = 400


class ZeusBSOServiceError(ZeusBSOServerBaseError):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.status_code = 500


class ZeusBSOServiceBadRequestError(ZeusBSOServerBaseError):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.status_code = 400
