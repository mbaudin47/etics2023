# %%
import openturns as ot
import openturns.viewer as otv
import matplotlib.pyplot as plt

# %%
def printInterval(interval, indentation="   "):
    """
    Print the [g, h0] C.I. with readable units.
    """
    lowerBound = interval.getLowerBound()
    upperBound = interval.getUpperBound()
    print(
        indentation,
        "g in [%.3f, %.3f]" % (lowerBound[0] , upperBound[0] ),
    )
    print(
        indentation,
        "h0 in [%.2f, %.2f]" % (lowerBound[1], upperBound[1] ),
    )
    return None

# %%
def plotDistributionGridPDF(distribution):
    """
    Plot the marginal and bi-dimensional iso-PDF on a grid.

    Parameters
    ----------
    distribution : ot.Distribution
        The distribution.

    Returns
    -------
    grid : ot.GridLayout(dimension, dimension)
        The grid of plots.

    """
    dimension = distribution.getDimension()
    grid = ot.GridLayout(dimension, dimension)
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                distributionI = distribution.getMarginal([i])
                graph = distributionI.drawPDF()
            else:
                distributionJI = distribution.getMarginal([j, i])
                graph = distributionJI.drawPDF()
            graph.setLegends([""])
            graph.setTitle("")
            if i < dimension - 1:
                graph.setXTitle("")
            if j > 0:
                graph.setYTitle("")
            grid.setGraph(i, j, graph)
    grid.setTitle("Iso-PDF values")
    return grid

# %%
data = ot.Sample.ImportFromCSVFile("Ball_drops_data.csv")
drop1 = data[0:51]
drop2 = data[51:]
drop1[:5]

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
print("+ Calibration with non linear least squares")
algo = ot.NonLinearLeastSquaresCalibration(
    dropBallModelParametric, timeSample, heightSample, thetaInitial
    )
algo.run()
calibrationResult = algo.getResult()
thetaMAP = calibrationResult.getParameterMAP()
print("MAP =", thetaMAP)

# %%
graph = calibrationResult.drawObservationsVsInputs()
aEstimated, bEstimated = thetaMAP
title = "After calibration : a = %.4f, b = %.4f" % (aEstimated, bEstimated)
graph.setTitle(title)
graph.setLegendPosition("topright")
view = otv.View(graph)

# %%
graph = calibrationResult.drawResiduals()
graph.setLegendPosition("topright")
view = otv.View(graph)

# %%
graph = calibrationResult.drawResidualsNormalPlot()
view = otv.View(graph)

# %%
graph = calibrationResult.drawParameterDistributions()
view = otv.View(
    graph,
    figure_kw={"figsize": (10.0, 4.0)},
    legend_kw={"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"},
)
plt.subplots_adjust(right=0.8)

# %%
thetaPosterior = calibrationResult.getParameterPosterior()
interval = thetaPosterior.computeBilateralConfidenceIntervalWithMarginalProbability(
    0.95
)[0]
print("95% C.I.:")
printInterval(interval)


# %%
grid = plotDistributionGridPDF(thetaPosterior)
view = otv.View(
    grid,
    figure_kw={"figsize": (6.0, 6.0)},
    legend_kw={"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"},
)
plot_space = 0.5
plt.subplots_adjust(wspace=plot_space, hspace=plot_space)

# %%
print("+ Gaussian calibration")
sigmaHeight = 1.e-1  # (m)

# %%
errorCovariance = ot.CovarianceMatrix(1)
errorCovariance[0, 0] = sigmaHeight ** 2

# %%
sigmaG = 1
sigmaH0 = 5
covarianceMatrix = ot.CovarianceMatrix(2)
covarianceMatrix[0, 0] = sigmaG ** 2
covarianceMatrix[1, 1] = sigmaH0 ** 2
print("Prior covariance matrix")
print(covarianceMatrix)

# %%
algo = ot.GaussianLinearCalibration(
    dropBallModelParametric, timeSample, heightSample, thetaInitial, 
    covarianceMatrix, errorCovariance
)
algo.run()
calibrationResult = algo.getResult()
print("MAP =", thetaMAP)
thetaPosterior = calibrationResult.getParameterPosterior()
interval = thetaPosterior.computeBilateralConfidenceIntervalWithMarginalProbability(
    0.95
)[0]
print("95% C.I.:")
printInterval(interval)

# %%
graph = calibrationResult.drawObservationsVsPredictions()
view = otv.View(graph)

# %%
graph = calibrationResult.drawParameterDistributions()
view = otv.View(
    graph,
    figure_kw={"figsize": (10.0, 4.0)},
    legend_kw={"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"},
)
plt.subplots_adjust(right=0.8)

# %%
