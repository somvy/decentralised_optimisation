from torch import Tensor


class BaseOracle:
    def __call__(self):
        raise NotImplementedError('Oracle call is not implemented!')

    def set_params(self, params: list[Tensor]):
        raise NotImplementedError('set_params is not implemented!')

    def get_params(self) -> list[Tensor]:
        """return detached params"""
        raise NotImplementedError('get_params is not implemented!')

    def grad(self) -> list[Tensor]:
        """length of the list should be equal to len of get_params()"""
        raise NotImplementedError('Oracle gradient is not implemented!')

    def subgrad(self):
        raise NotImplementedError('Oracle subgradient is not implemented!')
