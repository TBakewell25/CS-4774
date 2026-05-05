import pandas as pd
import sys


def main(args):
    df = pd.read_csv(args[1]).head()

    df.to_csv("example.csv",index=False)

    return


if __name__=="__main__":
    main(sys.argv)
