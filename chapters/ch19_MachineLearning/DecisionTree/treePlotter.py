# treePlotter.py (Python 3 compatible)
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode     = dict(boxstyle="round4",   fc="0.8")
arrow_args   = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    firstStr   = next(iter(myTree))
    secondDict = myTree[firstStr]
    leafs = 0
    for _, val in secondDict.items():
        if isinstance(val, dict):
            leafs += getNumLeafs(val)
        else:
            leafs += 1
    return leafs

def getTreeDepth(myTree):
    firstStr   = next(iter(myTree))
    secondDict = myTree[firstStr]
    maxDepth = 0
    for _, val in secondDict.items():
        if isinstance(val, dict):
            thisDepth = 1 + getTreeDepth(val)
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(
        nodeTxt, xy=parentPt, xycoords='axes fraction',
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args
    )

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs  = getNumLeafs(myTree)
    firstStr  = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key, val in secondDict.items():
        if isinstance(val, dict):
            plotTree(val, cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(str(val), (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white', figsize=(10, 6))
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
    # If you're in a headless environment and nothing pops up, add:
    plt.savefig("decision_tree.png", bbox_inches="tight")
