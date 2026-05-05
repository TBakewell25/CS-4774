import pandas as pd
import sys


def main(args):
    df = pd.read_csv(args[1])

    col_max = df["CycleNumber"].max()

    print(col_max)


    return

if __name__=="__main__":
    main(sys.argv)

    
