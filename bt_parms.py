from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types
# spec = [
#     ('res', float32),               # a simple scalar field
# ]
#
# @jitclass(spec)
class bt_parms(object):

    def __init__(self, res):
        self.res = res

    @property
    def fs(self):
        return 44100

    @property
    def timeres(self):
        return (round(self.fs * self.res))

    @property
    def winlen(self):
        return int(round((512 ** 2) / self.timeres))

    @property
    def step(self):
        return int(round(self.winlen / 4))

    @property
    def bwinlen(self):
        return 512

    @property
    def bstep(self):
        return 128

    @property
    def stepthresh(self):
        return int(round(8 * (512 / self.timeres)))

    # @property
    # def constthresh(self):
    #     return int(round(self.stepthresh() / 2))

    @property
    def rayparam(self):
        return round(43 * (512 / self.timeres))

    @property
    def pmax(self):
        return int(round(120 * (512 / self.timeres)))

    @property
    def pmin(self):
        return int(round(4 * (512 / self.timeres)))

    @property
    def lowest(self):
        return int(round(21 * (512 / self.timeres)))

    @property
    def pre(self):
        return int(round(3 * (512 / self.timeres)))

    # @property
    # def post(self):
    #     return self.pre()

    @property
    def fact(self):
        return 60 * self.fs / self.timeres


