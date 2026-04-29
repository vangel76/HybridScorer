class SkipValue:
    pass


SKIP = SkipValue()


class Update(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.__dict__.update(kwargs)


def update(**kwargs):
    return Update(**kwargs)


def skip():
    return SKIP


class Progress:
    def __call__(self, value=0, desc=None):
        return None


class SelectData:
    def __init__(self, index=None):
        self.index = index
