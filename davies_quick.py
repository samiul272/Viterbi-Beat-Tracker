from periodicity_path import periodicity_path
from dynamic_programming import dynamic_programming
from bt_parms import bt_parms


def davies_quick(df, p=bt_parms, mode=0):
    # strip any trailing zeros
    while df[-1] == 0:
        df = df[0:len(df) - 1]

    # get periodicity path
    ppath = periodicity_path(df, p)

    # find beat locations
    beats = dynamic_programming(df, p, ppath, mode)


def davies_quick2(df, p, alpha, tightness, twidth):
    while df[-1] == 0:
        df = df[0:len(df) - 1]

    # get periodicity path
    ppath = periodicity_path2(df, p, twidth)
    mode = 0  # use this to run normal algorithm

    # find beat locations
    beats = dynamic_programming2(df, p, alpha, tightness)
