


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


df.groupby(['ring-number', 'class']).size().unstack().plot.bar(stacked=True)

