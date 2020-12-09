#!/usr/bin/python

import csv
import sys
import os 
import os.path
import urllib.request


#### Implementation that uses local/manually fetched data, might be stale

# You might have to change this if you're running in another directory! 
CASES_FILE = 'cases_timeseries_canada.csv'


def num_cases_by_day(date):
    # print(os.path.gettime(CASES_FILE))
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

# ex. calling this prints "This won't be a full 7-day week!" the value
num_cases_by_week("17-11-2020")
# also yikes Canada isn't doing too hot






#### Implementation that fetches data every time, fresh, might be slow

def num_cases_by_day_fresh(date):
    url = "https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_canada/cases_timeseries_canada.csv"
    file = urllib.request.urlopen(url)
    lines = [l.decode('utf-8') for l in file.readlines()]
    reader = csv.reader(lines)

    for row in reader:
        if row[1] == date: 
            print(str(row[2]))
            return row[2]

# for example, calling this prints/returns 211
num_cases_by_day_fresh("03-08-2020")

def num_cases_by_week_fresh(first_day_of_week):    
    url = "https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_canada/cases_timeseries_canada.csv"
    file = urllib.request.urlopen(url)
    lines = [l.decode('utf-8') for l in file.readlines()]
    reader = csv.reader(lines)

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
num_cases_by_week_fresh("11-03-2020")

# ex. calling this prints "This won't be a full 7-day week!" and the value
num_cases_by_week_fresh("22-11-2020")
# also yikes Canada isn't doing too hot

