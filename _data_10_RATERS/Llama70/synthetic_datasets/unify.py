import pandas as pd
synthetic_1 = pd.read_csv("synthetic_ME_" + str(1) + ".csv")
synthetic_2 = pd.read_csv("synthetic_ME_" + str(2) + ".csv")
synthetic = pd.concat([synthetic_1, synthetic_2])
synthetic.to_csv("synthetic_MM" + ".csv", index = False)