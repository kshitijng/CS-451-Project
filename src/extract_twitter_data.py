import os
import json
import csv
import ast
import heapq
import sys
from datetime import timedelta, date, datetime
csv.field_size_limit(sys.maxsize)

INPUT = "tst_result"

KEY_TERMS = {
    'global_bigrams': ['SOCIAL DISTANCING', 'COVID19 PANDEMIC',
                       'COVID 19', 'CASES OF', 'THE PANDEMIC', 'COVID19 CASES',
                       'CORONAVIRUS CASES', 'NEW CASES', 'SPREAD OF',
                       'TO WEAR', 'COVID19 TESTING', 'FACE MASKS',
                       'TESTED POSITIVE', 'A PANDEMIC', 'COVID19 IN',
                       'CORONAVIRUS IN', 'THE VIRUS'],
    'global_unigrams': ['DEATHS', 'POSITIVE', 'RESTRICTIONS', 'DEATH', 'CRISIS',
                        'CORONA'],
    'canada_bigrams': ['COVID19 CASES', 'NEW COVID19', 'COVID CASES',
                       'JOHNS HOPKINS', 'COVID19 PANDEMIC', 'CORONAVIRUS CASES',
                       'NEW COVID', 'SURGED TO', 'CANADA COVID19',
                       'CASES TODAY', 'WAVE OF', 'SPREAD OF',
                       'HAVE INCREASED', 'TESTED POSITIVE', 'SOCIAL DISTANCING'],
    'canada_unigrams': ['COVID19', 'COVID', 'CASES', 'NEW', 'PANDEMIC',
                        'ONTARIO', 'TORONTO', 'CORONAVIRUS', 'HOPKINS', 'CASE',
                        'OTTAWA', 'LOCKDOWN', 'TESTS', 'TESTING', 'QUARANTINE'
                        'POSITIVE', 'DEATHS', 'DEATH', 'CORONA']
}

TYPE_SHORT = {'canada_unigrams': 'CU', 'canada_bigrams': 'CB',
              'global_unigrams': 'GU', 'global_bigrams': 'GB'}

OUTPUT = 'data/twitter'
MONTHS = ['04_apr', '05_may', '06_jun', '07_jul', '08_aug', '09_sep', '10_oct', '11_nov']

def output_data(type, data):
    dates = sorted(data.keys())

    for term in KEY_TERMS[type]:
        term_output = '-'.join(term.lower().split(" "))
        filename = "{}_{}".format(TYPE_SHORT[type], term_output)
        outfile = os.path.join(OUTPUT, filename)

        with open(outfile, 'w', newline='') as csvfile:
            w = csv.writer(csvfile, delimiter=',',
                           quotechar='|', quoting=csv.QUOTE_MINIMAL)

            w.writerow(['date', filename])
            for date in dates:
                w.writerow([date, data[date].get(term, 0)])



if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)

for k in KEY_TERMS:
    print("Doing: {}".format(k))
    type_data = {}

    for month in MONTHS:
        print("  Month: {}".format(month))

        month_dir = os.path.join(INPUT, k, month)

        month_files = os.listdir(month_dir)
        month_filename = ''

        for f in month_files:
            if f[:4] == "part":
                month_filename = f

        month_file = os.path.join(month_dir, month_filename)

        with open(month_file, 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                try:
                    row_date = datetime.strptime(row[0], "%b%d")
                except:
                    print("    ERROR WITH: " + str(row))
                    print("    SKIPPING ROW...")
                    continue
                row_date_out = datetime.strftime(row_date, "2020-%m-%d")

                if row_date_out not in type_data:
                    type_data[row_date_out] = {}

                day_data = ast.literal_eval(row[1])

                for term in KEY_TERMS[k]:
                    type_data[row_date_out][term] = day_data.get(term, 0) + \
                        type_data[row_date_out].get(term,0)


    print("  Outputting type data")
    output_data(k, type_data)
    print("  Done")
