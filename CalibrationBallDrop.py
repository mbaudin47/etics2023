# %%
import openturns as ot

# %%
data = ot.Sample.ImportFromCSVFile("Ball_drops_data.csv")
data[:5]

# %%
def dropBallModel(x):
    g = x[0]
    h0 = x[1]
    t = x[2]
    h = -g * 0.5 * t ** 2 + h0
    h = max(0.0, h)
    return h
