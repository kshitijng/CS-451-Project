#!/usr/bin/python

import csv
import sys
from os import listdir
from os.path import isfile, join

# You might have to change this if you're running in another directory! 
CASES_FILE = 'cases_timeseries_canada.csv'


def num_cases_by_day(date):
    with open(CASES_FILE, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[1] == date: 
                print(str(row[2]))
                return row[2]

# for example, calling this prints/returns 211
num_cases_by_day("03-08-2020")

def num_cases_by_week(first_day_of_week):
    with open(CASES_FILE, 'r') as file:
        reader = csv.reader(file)
        counting = 0
        for row in reader:
            if row[1] == first_day_of_week: 
                counting += int(row[2])
                for _ in range(6):
                    try:
                        row = next(reader)
                        counting += int(row[2])
                    except: 
                        print("This won't be a full 7-day week!")
                        break
                print(counting)
                return counting
        print("date out of range")
        return -1

# ex. calling this prints/returns 502
num_cases_by_week("11-03-2020")

# ex. calling this prints "This won't be a full 7-day week!" and prints/returns 20369
num_cases_by_week("15-11-2020")
# also yikes Canada isn't doing too hot

