# %%
import openturns as ot

# %%
data = ot.Sample.ImportFromCSVFile("Ball_drops_data.csv")
data[:5]

# %%
def dropBallModelPy(x):
    g = x[0]
    h0 = x[1]
    t = x[2]
    h = -g * 0.5 * t ** 2 + h0
    h = max(0.0, h)
    return [h]

# %%
theta=[9.8, 46]
dropBallModel = ot.PythonFunction(3, 1, dropBallModelPy)
dropBallModel.setInputDescription(["g", "h0", "t"])
dropBallModel.setOutputDescription(["h"])
dropBallModelParametric = ot.ParametricFunction(dropBallModel, [0, 1], theta)
dropBallModelParametric

# %%
graph = dropBallModelParametric.draw(0.0, 3.0)
graph


# %%
