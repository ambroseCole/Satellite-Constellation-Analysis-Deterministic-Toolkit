import math
from sgp4.api import Satrec, WGS72
from sgp4.api import jday

mu = 398600.4418

jd, fr = jday(2026, 5, 28, 0, 0, 0.0)
epoch = (jd + fr) - 2433281.5

class WalkerDeltaSGP:
    

    def __init__(self, argConstellation):

        self.sma = float(argConstellation['sma'])

        self.inc = float(argConstellation['inc'])

        self.firstRAAN = float(argConstellation['firstRAAN'])

        self.t = int(argConstellation['t'])
        self.p = int(argConstellation['p'])
        self.f = int(argConstellation['f'])

    def generate(self):
        satellites = []

        s = self.t/self.p
        if s.is_integer():
            s = int(s)
        else:
            raise(ValueError('Number of satellites per plane (t/p) should be integer'))

        for idxP in range(self.p):
            raan = self.firstRAAN + idxP * 360./self.p
            for idxS in range(s):
                meanAnomaly = idxP * self.f * 360./ self.t + idxS * 360./ s

                a_km = self.sma / 1000.0
                n_rad_s = math.sqrt(mu / a_km**3)
                no_kozai = n_rad_s * 60.0

                satCur = Satrec()
                satCur.sgp4init(
                    WGS72,
                    'i',
                    90000+s*idxP+idxS,
                    epoch,
                    2e-5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    self.inc*math.pi/180,
                    meanAnomaly*math.pi/180,
                    no_kozai,
                    raan*math.pi/180
                )

                satellites.append(satCur)

        return satellites


if __name__ == "__main__":
    import numpy as np
    argConst = {
        'sma': 6921000.,  # 550 km altitude
        'inc': 53.,
        'firstRAAN': 0.,
        't': 72 * 22,  # 1584 sats
        'p': 72,
        'f': 1
    }
    wc = WalkerDeltaSGP(argConst)
    sats = wc.generate()
    print(f"Generated {len(sats)} satellites in {argConst['p']} planes")