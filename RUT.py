from Task import Task
import argparse





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str)
    parser.add_argument('-m',type=str,default=None)
    parser.add_argument('-p', type=str, default=None)
    args = parser.parse_args()
    # print(args.m)