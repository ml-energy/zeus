"""Zeus batch size optimizer client exceptions."""

from zeus.exception import ZeusBaseError


class ZeusBSORuntimError(ZeusBaseError):
    """This error will be raised when the bso server failed to process the request correctly"""

    pass


class ZeusBSOTrainFailError(ZeusBaseError):
    """This error will be raised when the training is failed for the chosen batch_size"""

    pass


class ZeusBSOConfigError(ZeusBaseError):
    """This error will be raised when the configuration of training doesn't meet the requirements. ex) heterogeneous GPU"""

    pass


class ZeusBSOOperationOrderError(ZeusBaseError):
    """This error will be raised when the order of calling methods of BatchSizeOptimizer is wrong"""

    pass
