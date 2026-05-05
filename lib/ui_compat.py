class SkipValue:
    pass


SKIP = SkipValue()


class Update(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


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
