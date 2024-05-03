"""Zeus batch size optimizer client exceptions."""

from zeus.exception import ZeusBaseError


class ZeusBSORuntimError(ZeusBaseError):
    """Bso server failed to process the request correctly."""

    pass


class ZeusBSOTrainFailError(ZeusBaseError):
    """Training failed for the chosen batch_size."""

    pass


class ZeusBSOConfigError(ZeusBaseError):
    """Configuration of training doesn't meet the requirements. ex) heterogeneous GPU."""

    pass


class ZeusBSOOperationOrderError(ZeusBaseError):
    """Order of calling methods of BatchSizeOptimizer is wrong."""

    pass


class ZeusBSOBadOperationError(ZeusBaseError):
    """The usage of operations is wrong."""

    pass
