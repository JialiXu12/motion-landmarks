class Params(object):

    def __init__(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class FilterParams(Params):
    def __init__(self, path=None):
        super(FilterParams, self).__init__()
        self.crop = [[0, -1], [0, -1], [0, -1]]
        self.jobs = 1

        if path is not None:
            self.load(path)


class FitParams(Params):
    def __init__(self, path=None):
        super(FitParams, self).__init__()
        if path is not None:
            self.load(path)
