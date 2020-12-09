You can run the get_covid_case_data.py helper functions with hardcoded examples 
by doing `python3 get_covid_case_data.py`

I got this covid case data from this repo: https://github.com/ishaberry/Covid19Canada
File: https://github.com/ishaberry/Covid19Canada/blob/master/timeseries_canada/cases_timeseries_canada.csv


If you're having trouble with ssl stuff and you're on macOS you might have to 
do the following: 
Go to Macintosh HD > Applications > Python3.6 folder (or whatever version of python3 you're using) > double click on "Install Certificates.command" file.







If we want more data I also browsed Kaggle for some case-related data and we can flesh it out more if we want 
https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset?select=covid_19_data.csv
- Has to be filtered for just Canada, inconsistent date format, doesn't report for every day like the ishaberry dataset does so more complicated scripting necessary 
https://www.kaggle.com/skylord/coronawhy
- Cute name
- Just Canadian data
- Also doesn't report for every day, but has a `report_week` column (values are dates 7 days apart on Sundays) so we could probably easily search for non-sliding window dates if we wanted this data. 

