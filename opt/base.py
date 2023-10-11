

class BaseNode:
    def __init__(self, fun, x):
        """
        :param fun:
        fun should implement __call__ and return a scalar
        and first order gradient via .grad()
        or subgradient via .subgrad()

        :param x:
        """
        self.fun = fun
        self.x = x

    def step(self):
        pass
