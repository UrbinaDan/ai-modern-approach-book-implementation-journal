#This code implements id3 and c4.5 decision tree algorithms
#and visualizes the resulting tree using treePlotter.py

from id3_c45 import DecisionTree          # <-- your class file
import numpy as np
import treePlotter                        # <-- matches treePlotter.py


if __name__ == "__main__":
    import numpy as np

    # toy dataset (categorical + numeric)
    X = np.array([
        ["AltNo","BarNo","Fri","Hungry","Some", "$$", "NoRain","NoRes","Thai",  "10-30"],
        ["AltYes","BarYes","Sat","Hungry","Full","$$$","Rain","NoRes","French", ">60"],
        ["AltNo","BarYes","Sun","NotHungry","None","$", "NoRain","Res","Thai",  "0-10"],
    ], dtype=object)

    y = np.array(["Yes","No","Yes"], dtype=object)

    feat_names = ["Alt","Bar","FriSat","Hungry","Patrons","Price",
                  "Raining","Reservation","Type","WaitEst"]

    clf = DecisionTree(criterion="C4.5", feature_names=feat_names, verbose=True).fit(X, y)
    print("Pred on first row:", clf.predict(X[0]))
    # visualize (requires treePlotter.py with createPlot)
    clf.show()