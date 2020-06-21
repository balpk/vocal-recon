import os
from os import listdir
from os.path import isfile, join
import re
import  pandas as pd
import numpy as np

def main():
    recordings = [11, 51]
    base_path="../plots/"
    recording_nrs = []
    identifiers = []
    outfile = "all_identifiers.csv"
    # find all files of form seq_[nr]_mic and extract that number; store this with rec nr as a .csv
    for rec in recordings:
        path = os.path.join(base_path, "rec%02d/" % (rec))
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for f in files:
            if f.endswith("mic.png"):
                s = re.findall("seq_([0-9][0-9][0-9][0-9])_mic.png", f)
                assert len(s) == 1
                recording_nrs.append(rec)
                identifiers.append("seq_" + s[0])

    df = pd.DataFrame(data=np.array([recording_nrs, identifiers]).transpose())
    df.columns=["Recording Nr", "Sequence ID"]
    df = df.sort_values(by=["Recording Nr", "Sequence ID"], axis=0)
    df.to_csv(outfile, sep=",", )

if __name__ == '__main__':
    main()