from torch import tensor


class BaseOracle:
    def __call__(self, x):
        raise NotImplementedError('Oracle call is not implemented!')

    def set_params(self, params: list[tensor]):
        raise NotImplementedError('set_params is not implemented!')

    def get_params(self) -> list[tensor]:
        """return detached params"""
        raise NotImplementedError('get_params is not implemented!')

    def grad(self, x) -> list[tensor]:
        """length of the list should be equal to len of get_params()"""
        raise NotImplementedError('Oracle gradient is not implemented!')

    def subgrad(self, x):
        raise NotImplementedError('Oracle subgradient is not implemented!')
