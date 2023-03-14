c_intermitted = 0.189 / 0.58 + 0.811 / 0.91
c_shi = 1 / 0.895

class Model:
    def __init__(self, fun, fids, pids):
        # FIXME: fids and pids could be dictionaries (key-pid/fid, value-index)
        self.fun = fun
        self.fids = fids
        self.pids = pids

def model_extended(t, y, p):
    y_art, y_prep = y
    k_art, k_prep, n_msm = p
    f_art = k_art * y_art
    f_prep = k_prep * (n_msm - c_intermitted * c_shi * y_prep)
    return f_art, f_prep

