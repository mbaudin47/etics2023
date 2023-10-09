# %%
import openturns as ot
import openturns.viewer as otv

# %%
data = ot.Sample.ImportFromCSVFile("Ball_drops_data.csv")
data[:5]

# %%
timeSample = data[:, 1]
heightSample = data[:, 2]
timeStart = timeSample[0, 0]
timeStop = timeSample[-1, 0]

# %%
def dropBallModelPy(x):
    g = x[0]
    h0 = x[1]
    t = x[2]
    h = -g * 0.5 * t ** 2 + h0
    h = max(0.0, h)
    return [h]

# %%
gInitial = 9.8
h0Initial = 46.0
thetaInitial=[gInitial, h0Initial]
dropBallModel = ot.PythonFunction(3, 1, dropBallModelPy)
dropBallModel.setInputDescription(["g", "h0", "t"])
dropBallModel.setOutputDescription(["h"])
dropBallModelParametric = ot.ParametricFunction(dropBallModel, [0, 1], thetaInitial)
dropBallModelParametric

# %%
graph = dropBallModelParametric.draw(timeStart, timeStop)
graph


# %%
title = "Before calibration : a = %.4f, b = %.4f" % (gInitial, h0Initial)
graph = ot.Graph(title, "Q", "H", True)
# Plot the model before calibration
curve = dropBallModelParametric.draw(timeStart, timeStop).getDrawable(0)
curve.setLegend("Model, before calibration")
graph.add(curve)
# Plot the noisy observations
cloud = ot.Cloud(timeSample, heightSample)
cloud.setLegend("Observations")
cloud.setPointStyle("bullet")
graph.add(cloud)
#
graph.setColors(ot.Drawable.BuildDefaultPalette(2))
graph.setLegendPosition("topright")
view = otv.View(graph)

# %%
# Calibration with non linear least squares
algo = ot.NonLinearLeastSquaresCalibration(
    dropBallModelParametric, timeSample, heightSample, thetaInitial
    )
algo.run()
calibrationResult = algo.getResult()
thetaMAP = calibrationResult.getParameterMAP()
print(thetaMAP)

# %%
