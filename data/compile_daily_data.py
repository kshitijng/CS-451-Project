""" Prepare the csv that will be input to the ML model

Since Twitter Data starts on April 1, we can prepare data from April 8, onwards
We can prepare data for up to 7 days before present.

The CSV will have the following fields:
    1. Date (ex. 2020-11-20)
    2. Actual cases in the week starting on date (2020-11-20 to 2020-11-26)
    3. Number of new cases in the last week (2020-11-13 to 2020-11-19)
    4. Number of new cases in the week before last (2020-11-06 to 2020-11-12)
    5. Active Cases on the previous day (2020-11-19)
    6. Google trend over the last week for "anti-mask" (2020-11-13 to 2020-11-19)
    7. Google trend over the last week for "coronaivirus"
    8. Google trend over the last week for "cough"
    9. Google trend over the last week for "covid"
    10. Google trend over the last week for "fatigue"
    11. Google trend over the last week for "flu"
    12. Google trend over the last week for "headache"
    13. Google trend over the last week for "lockdown"
    14. Google trend over the last week for "mask"
    15. Google trend over the last week for "sick"
    16. Google trend over the last week for "symptoms"
    17. Google trend over the last week for "tired"
    18. Google trend over the last week for "virus"
    19 - ??. Results from the Twitter data analysis over the last week (2020-11-13 to 2020-11-19)
"""

import csv
import sys
from os import listdir
from os.path import isfile, join
from datetime import timedelta, date, datetime

GOOGLE_DATA = 'google_trends'
NEW_CASES_DATA = 'covid_cases/cases_timeseries_canada.csv'
ACTIVE_CASES_DATA = 'covid_cases/active_timeseries_canada.csv'
OUTPUT_FILE = 'ml_input.csv'


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def add_col(full_data, new_col):
    for i in range(len(new_col)):
        full_data[i].append(new_col[i])


def read_csv_data(infile, start_date, end_date, date_col=0, data_col=1):
    data = []
    reading = False
    with open(infile, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            if row[date_col] == start_date:
                reading = True
            if row[date_col] == end_date:
                break
            if reading:
                data.append(int(row[data_col]))

    return data


def get_covid_data(start_date, end_date, full_data):
    # Handle new case data (2 weeks ago, last week, week starting on date)
    print("Processing new case data")
    start_date = start_date - timedelta(days=14)
    c_data = read_csv_data(infile=NEW_CASES_DATA,
                           start_date=start_date.strftime("%d-%m-%Y"),
                           end_date=datetime.today().strftime("%d-%m-%Y"),
                           date_col=1,
                           data_col=2)

    prev2 = ["new_cases_2w_ago"]
    # prev2sum = sum(c_data[0:7])
    prev = ["new_cases_last_week"]
    # prevsum = sum(c_data[7:14])
    next = ["actual_cases"]
    # nextsum = sum(c_data[14:21])

    for i in range(len(c_data) - 20):
        prev2.append(sum(c_data[i:i+7]))
        prev.append(sum(c_data[i+7:i+14]))
        next.append(sum(c_data[i+14:i+21]))

    add_col(full_data, next)
    add_col(full_data, prev)
    add_col(full_data, prev2)

    # Handle active case data
    print("Processing active case data")
    start_date = start_date + timedelta(days=13)
    a_data = read_csv_data(infile=ACTIVE_CASES_DATA,
                           start_date=start_date.strftime("%d-%m-%Y"),
                           end_date=end_date.strftime("%d-%m-%Y"),
                           date_col=1,
                           data_col=5)

    a_data = ["active_cases"] + a_data
    add_col(full_data, a_data)


def get_google_data_files(dir):
    files = []
    for f in listdir(dir):
        if isfile(join(dir, f)) and f.split('.')[-1] == 'csv':
            files.append(f)
    return files


def get_google_data(start_date, end_date, full_data):
    files = get_google_data_files(GOOGLE_DATA)
    start_date = start_date - timedelta(days=7)

    for f in sorted(files):
        print("Processing {}".format(f))
        col = ["G_{}".format(f.split('_')[0])]
        data = read_csv_data(join(GOOGLE_DATA, f),
                             start_date.strftime("%Y-%m-%d"),
                             end_date.strftime("%Y-%m-%d"))

        val = sum(data[:7])
        col.append(val)
        for i in range(len(data) - 7):
            val -= data[i]
            val += data[i+7]
            col.append(val)

        add_col(full_data, col)


def output_data(full_data):
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for r in full_data:
            w.writerow(r)


def main():
    start_date = datetime.strptime("2020-04-08", "%Y-%m-%d")
    # Set to six days ago (Iterator skips the last day)
    end_date = datetime.today() - timedelta(days=6)

    full_data = [["date"]]

    for d in daterange(start_date, end_date):
        full_data.append([d.strftime("%Y-%m-%d")])

    end_date = end_date - timedelta(days=1)

    get_covid_data(start_date, end_date, full_data)

    get_google_data(start_date, end_date, full_data)

    output_data(full_data)


if __name__ == "__main__":
    main()
