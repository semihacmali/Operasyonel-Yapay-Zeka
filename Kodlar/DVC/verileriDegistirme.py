# -*- coding: utf-8 -*-
"""
@author: SemihAcmali
"""

import pandas as pd

data = pd.read_csv("deneyim_maas.csv")

print(data.head)

data["deneyim_yili"] = 2 * data["deneyim_yili"]

data.to_csv("deneyim_maas.csv", index = False)
