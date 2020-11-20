"""
Run as `python3 batch_by_month.py "apr,may,jun,jul,aug,sep,oct,nov"`
The value passed in is a comma-separated list of the months we want to aggregate for
We can just use nov going forward.
"""

import csv
import sys
from os import listdir
from os.path import isfile, join

RAW_DATA = 'ieee_raw'
AGGREGATED_OUTPUT = 'ieee_raw_monthly'
# MONTHS = ['april','may','june','july','august','september','october','november']

if __name__ == "__main__":
    months = sys.argv[1].split(",")

    files = [f for f in listdir(RAW_DATA) if isfile(join(RAW_DATA, f))]
    print("Found {} files".format(len(files)))
    print("Working on {} months ({})".format(len(months),sys.argv[1]))

    for month in months:
        # print("\nOn month {}".format(month))
        output_file = join(AGGREGATED_OUTPUT, '{}.txt'.format(month))

        with open(output_file, 'w') as out_f:
            files_found = 0
            tweets_found = 0
            for f in files:
                # print(".")

                if f[:3] == month[:3]:
                    files_found += 1
                    # print("Matched {} with {}".format(month,f))
                    with open(join(RAW_DATA, f), 'r') as in_f:
                        spamreader = csv.reader(in_f, delimiter=',', quotechar='|')
                        for row in spamreader:
                            tweets_found += 1
                            out_f.write("{}\n".format(row[0]))

            print("\n{} - Found {} files with {} tweets".format(month,files_found,tweets_found))
