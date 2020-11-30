import os
import json
import csv
import ast
import heapq
import sys
csv.field_size_limit(sys.maxsize)

TOP_NUM = 1000

INPUT = "tst_result"
TYPES = ['global_bigrams', 'global_unigrams', 'canada_bigrams', 'canada_unigrams']
MONTHS = ['04_apr', '05_may', '06_jun', '07_jul', '08_aug', '09_sep', '10_oct', '11_nov']

OUTPUT = "data/popular"

def output_data(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for r in data:
            i += 1
            w.writerow([i, -r[0], r[1]])

if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)

for type in TYPES:
    print("Doing: {}".format(type))
    type_data = {}
    type_out = os.path.join(OUTPUT, type)
    if not os.path.exists(type_out):
        os.mkdir(type_out)

    for month in MONTHS:
        print("  Month: {}".format(month))
        month_data = {}

        month_dir = os.path.join(INPUT, type, month)

        month_files = os.listdir(month_dir)
        month_filename = ''

        for f in month_files:
            if f[:4] == "part":
                month_filename = f

        month_file = os.path.join(month_dir, month_filename)

        with open(month_file, 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                day_data = ast.literal_eval(row[1])
                for k, v in day_data.items():
                    month_data[k] = month_data.get(k, 0) + v

        h = []
        for k, v in month_data.items():
            type_data[k] = type_data.get(k, 0) + v
            heapq.heappush(h, (-v, k))

        sorted = [heapq.heappop(h) for i in range(TOP_NUM)]

        out_path = os.path.join(OUTPUT, type, month+".csv")
        output_data(sorted, out_path)

        print("    Processed {} Terms".format(len(month_data)))

    h2 = []
    for k, v in type_data.items():
        heapq.heappush(h2, (-v, k))

    sorted = [heapq.heappop(h2) for i in range(TOP_NUM)]

    out_path = os.path.join(OUTPUT, type, "COMPLETE.csv")
    output_data(sorted, out_path)
    print("  Processed {} Terms".format(len(type_data)))
