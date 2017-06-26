#!/usr/bin/env python3

import csv, json
import statistics as st
import datetime
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import re
from collections import OrderedDict
from os import path


rows = [] # hourly
daily = []
monthly = []

get_digit = r'\d+'


colors = ["red","green","blue","magenta","brown", "purple"]
_color_index = -1

def getColor():
    global _color_index
    _color_index += 1
    while _color_index >= len(colors): _color_index -= len(colors)

    return colors[_color_index]

def get_linear_date(date):
    return date[0]*365*24+date[1]*30*24+date[2]*24


def mk_int(s):
    try:
        return int(s) if s else 0
    except ValueError:
        digits = re.findall(r'\d+', s)
        return int(digits[0]) if len(digits)>0 else 0

def parse_date_time(string):
    _date,_time = string.split(' ')

    year,month,day = _date.split('-')
    try:
        hour,minute = _time.split(':')
    except ValueError:
        hour, minute = 0,0

    return (int(year),int(month),int(day), int(hour), int(minute))

def mt2min(_time):
    '''Military time to minutes.'''
    return int(_time[:2]) * 60 + int(_time[2:])

def min2str(minute):
    return '{} hours {} minutes'.format(int(minute/36), int(minute - int(minute/60)))

def convert():
    print("Converting CSV to JSON")
    global rows, FILE
    with open(path.join('data',FILE+'.csv')) as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [dict(x) for x in list(reader)]
        print('{} rows found.'.format(len(rows)))

        rows.sort(key=lambda x: parse_date_time(x['DATE']))

        print('Writing hourly data')
        with open(path.join('data',FILE+'.json'), 'w') as outfile:
            _rows = [{k:val for k,val in x.items() if not k.startswith('Monthly') and not k.startswith('DAILY')} for x in rows]
            json.dump(_rows, outfile)

        print('Writing daily data')
        # daily
        with open(path.join(FILE+'.daily.json'), 'w') as outfile:
            _rows = [{k:val for k,val in x.items() if not k.startswith('HOURLY') and not k.startswith('Monthly')} for x in rows]
            _days = set([parse_date_time(x['DATE'])[:3] for x in _rows])
            print('{} days found'.format(len(_days)))
            _rows = [
                    (_day, next(x for x in _rows if _day == parse_date_time(x['DATE'])[:3]))
                    for _day in _days]
            _rows.sort()
            json.dump(_rows, outfile)



        print('Writing monthly data')
        # monthly
        with open(path.join('data',FILE+'.monthly.json'), 'w') as outfile:
            _rows = [{k:val for k,val in x.items() if not k.startswith('HOURLY') and not k.startswith('DAILY')} for x in rows]
            _months = set([parse_date_time(x['DATE'])[:2] for x in _rows])
            print('{} months found'.format(len(_months)))
            _rows = [
                    (_month, next(x for x in _rows if _month == parse_date_time(x['DATE'])[:2]))
                    for _month in _months]
            _rows.sort()
            json.dump(_rows, outfile)

        print('raw data successfully written')

        # get daily data and write them
        # get monthly data and write them

    pass

def read(load_hourly=True, load_daily=True, load_monthly=True):
    print("Reading JSON")
    global rows, daily, monthly, FILE
    if load_hourly:
        with open(path.join('data',FILE+'.json')) as f:
            rows = json.load(f)
            print('{} hourly rows read'.format(len(rows)))

    if load_daily:
        with open(path.join('data',FILE+'.daily.json')) as f:
            daily = json.load(f)
            print('{} daily rows read'.format(len(daily)))

    if load_monthly:
        with open(path.join(FILE+'.monthly.json')) as f:
            monthly = json.load(f)
            print('{} monthly rows read'.format(len(monthly)))



def sunrise_on_years(month=1,day=1):
    global rows
    parsed = list(set([(parse_date_time(x['DATE'])[0:3], mt2min(x['DAILYSunrise'])) for x in rows]))
    parsed = list(set([(x[0],y) for x,y in parsed if (x[1] == month and x[2] == day)]))
    parsed.sort()
    print(parsed)
    plt.plot(*list(zip(*parsed)))
    plt.show()

def wind_dir_vs_date():
    global rows

    parsed = list(set(
        [(parse_date_time(x['DATE'])[0:3], mk_int(x['HOURLYWindSpeed']),mk_int(x['HOURLYWindDirection'])) for x in rows]))
    parsed = sorted(parsed, key=lambda x: x[0])

    dataset = []
    for day, values in groupby(parsed, key=lambda x: x[0]):
        row = list(zip(*values))[1:]
        row = list(zip(*row))
        matrix = np.asarray(row, dtype='float64')

        # matrix has a shape of x * 2
        # now, we convert second value to radians.
        matrix[:, 1] *= (2. * math.pi / 360.)

        # Then we extract components
        matrix[:, 0], matrix[:, 1] = matrix[:, 0]* np.cos(matrix[:, 1]), matrix[:, 0] * np.sin(matrix[:, 1])

        # Then average all the rows. Now we have the average component values
        matrix = np.average(matrix, axis=0)

        assert(matrix.shape == (2, ))

        time= day[0]*365*24+day[1]*30*24+day[2]*24
        x,y = matrix[0],matrix[1]
        dataset.append((x,y,time) )

    draw_3d(dataset, 'x','y','time')

def wind_dir_vs_hour():
    global rows

    hours = [(hour, []) for hour in range(24)]
    #print([parse_date_time(x['DATE'])[4] for x in rows])

    [hours[parse_date_time(x['DATE'])[3]][1].append((mk_int(x['HOURLYWindSpeed']), mk_int(x['HOURLYWindDirection'])))
                for x in rows]

    datasets = []
    for hour, values in hours:
        # if hour != 1: continue
        dataset = []
        matrix = np.asarray(values, dtype='float64')

        # matrix has a shape of x * 2
        # now, we convert second value to radians.
        matrix[:, 1] *= (2. * math.pi / 360.)

        # Then we extract components
        matrix[:, 0], matrix[:, 1] = matrix[:, 0]* np.cos(matrix[:, 1]), matrix[:, 0] * np.sin(matrix[:, 1])

        [dataset.append((x,y,hour)) for x,y in matrix.tolist()]
        datasets.append(dataset)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for value in datasets:
        ax.plot_wireframe(
                [x[0] for x in value],
                [x[1] for x in value],
                [x[2] for x in value], color=getColor())

    ax.set_xlabel("y")
    ax.set_ylabel("X")
    ax.set_zlabel("Hour")

def wind_magnitude_vs_hour(key='HOURLYWindSpeed'):
    print('Using key {}'.format(key))
    global rows
    hours = [(hour, []) for hour in range(24)]

    num_rows = 0
    # filter non-empty values
    for x in rows:
        date = parse_date_time(x['DATE'])
        data = x[key].strip()

        if data is '':
            continue

        hours[date[3]][1].append(mk_int(data))
        num_rows += 1

    if num_rows == 0:
        print('dataset empty')
        return
    print('Rendering {} rows'.format(num_rows))

    #[hours[parse_date_time(x['DATE'])[3]][1].append(mk_int(x['HOURLYWindSpeed'])) for x in rows]
    points = []
    for hour, values in hours:
        #plt.scatter(list(range(len(values))), values)

        mean = st.mean(values)
        sd = st.variance(values)**.5

        print("hour={0}, mean={1:.4f}, s.d.={2:.4f}".format(hour, mean, sd))
        points.append((hour, mean, sd, 0))

    plt.errorbar(*list(zip(*points)))
    plt.xlabel('Hour of day, 0-23')
    plt.ylabel('Average Mean {}'.format(key))
    plt.title('Mean {} per hour of day. (Error bars equal 1 s.d.)'.format(key))
    plt.show()
    #draw_2d(dataset)

def wind_magnitude_vs_year_vs_hour(key='HOURLYWindSpeed', show=True):
    """Show change of wind magnitude vs hour of day in years."""
    print('Using key {}'.format(key))
    global rows
    hours = [{} for hour in range(24)]

    num_rows = 0
    # filter non-empty values
    for x in rows:
        date = parse_date_time(x['DATE'])
        data = x[key].strip()

        if data is '': continue
        if date[0] not in hours[date[3]]:
            hours[date[3]][date[0]] = []
        hours[date[3]][date[0]].append(mk_int(data))

        num_rows += 1

    if num_rows == 0:
        print('dataset empty')
        return

    print('Rendering {} rows'.format(num_rows))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for hour, years in enumerate(hours):
        points = []
        for year, values in years.items():
            #plt.scatter(list(range(len(values))), values)

            mean = st.mean(values)
            sd = st.variance(values)**.5

            #print("hour={0}, mean={1:.4f}, s.d.={2:.4f}".format(hour, mean, sd))
            points.append((hour, year, mean))

        x,y,z = list(zip(*points))
        ax.plot(x,y,z, label="{}".format(hour))

    ax.legend()
    ax.set_title('Key: {}'.format(key))
    if show:
        plt.show()

    #plt.title('Mean {} per hour of day. (Error bars equal 1 s.d.)'.format(key))
    #plt.show()
    #draw_2d(dataset)

def wind_magnitude_vs_day_of_year_per_hour(data):
    """Show the graphs of wind_magnitude vs day_of_year for significant hours."""

    key = data['key']

    print('Using key {}'.format(key))
    global rows
    hours = [{} for hour in range(24)]

    num_rows = 0
    # filter non-empty values
    for x in rows:
        date = parse_date_time(x['DATE'])
        day_of_year = datetime.datetime(*date).timetuple().tm_yday
        val = x[key].strip()

        if val is '': continue
        if day_of_year not in hours[date[3]]:
            hours[date[3]][day_of_year] = []

        hours[date[3]][day_of_year].append(mk_int(val))
        num_rows += 1

    if num_rows == 0:
        print('dataset empty')
        return
    print('Rendering {} rows'.format(num_rows))

    SHS = [0, 6, 10, 12, 13, 14, 16, 20, 23]
    #SHS = [0]

    for hour, day_of_years in enumerate(hours):
        if hour not in SHS: continue;

        fig = plt.figure()
        ax = fig.gca()

        points = []
        for day_of_year, values in day_of_years.items():
            #plt.scatter(list(range(len(values))), values)

            mean = st.mean(values)
            #sd = st.variance(values)**.5

            #print("hour={0}, mean={1:.4f}, s.d.={2:.4f}".format(hour, mean, sd))
            points.append((day_of_year, hour, mean))
        points.sort(key = lambda x: x[0])

        x,y,z = list(zip(*points))
        ax.plot(x, z)

        ax.legend()
        ax.set_title('{} at hour {} for day of year(1-366)'.format(data['name'], hour))
        ax.set_xlabel('hour of day')
        ax.set_ylabel('{} ({})'.format(data['name'], data['unit']))

    #plt.title('Mean {} per hour of day. (Error bars equal 1 s.d.)'.format(key))
    #plt.show()
    #draw_2d(dataset)

def wind_magnitude_vs_hour_vs_year(key='HOURLYWindSpeed', show=True):
    print('Using key {}'.format(key))
    global rows
    hours = [(hour, []) for hour in range(24)]
    years = {}

    num_rows = 0
    # filter non-empty values
    for x in rows:
        date = parse_date_time(x['DATE'])
        data = x[key].strip()

        if data is '': continue
        if date[0] not in years:
            years[date[0]] = deepcopy(hours)

        years[date[0]][date[3]][1].append(mk_int(data))
        num_rows += 1

    if num_rows == 0:
        print('dataset empty')
        return
    print('Rendering {} rows'.format(num_rows))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for year, hours in years.items():
        points = []
        for hour, values in hours:
            #plt.scatter(list(range(len(values))), values)

            mean = st.mean(values)
            sd = st.variance(values)**.5

            #print("hour={0}, mean={1:.4f}, s.d.={2:.4f}".format(hour, mean, sd))
            points.append((year, hour, mean))

        x,y,z = list(zip(*points))
        ax.plot(x,y,z, label="{}".format(hour))

    ax.legend()
    ax.set_title('Key: {}'.format(key))
    if show:
        plt.show()

    #plt.title('Mean {} per hour of day. (Error bars equal 1 s.d.)'.format(key))
    #plt.show()
    #draw_2d(dataset)


def wind_magnitude_vs_hour_vs_year(key='HOURLYWindSpeed', show=True):
    print('Using key {}'.format(key))
    global rows
    hours = [(hour, []) for hour in range(24)]
    years = {}

    num_rows = 0
    # filter non-empty values
    for x in rows:
        date = parse_date_time(x['DATE'])
        data = x[key].strip()

        if data is '': continue
        if date[0] not in years:
            years[date[0]] = deepcopy(hours)

        years[date[0]][date[3]][1].append(mk_int(data))
        num_rows += 1

    if num_rows == 0:
        print('dataset empty')
        return
    print('Rendering {} rows'.format(num_rows))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for year, hours in years.items():
        points = []
        for hour, values in hours:
            #plt.scatter(list(range(len(values))), values)

            mean = st.mean(values)
            sd = st.variance(values)**.5

            #print("hour={0}, mean={1:.4f}, s.d.={2:.4f}".format(hour, mean, sd))
            points.append((year, hour, mean))

        x,y,z = list(zip(*points))
        ax.plot(x,y,z, label="{}".format(hour))

    ax.legend()
    ax.set_title('Key: {}'.format(key))
    if show:
        plt.show()

    #plt.title('Mean {} per hour of day. (Error bars equal 1 s.d.)'.format(key))
    #plt.show()
    #draw_2d(dataset)

def wind_magnitude_vs_hour_vs_year(key='HOURLYWindSpeed', show=True):
    print('Using key {}'.format(key))
    global rows
    hours = [(hour, []) for hour in range(24)]
    years = {}

    num_rows = 0
    # filter non-empty values
    for x in rows:
        date = parse_date_time(x['DATE'])
        data = x[key].strip()

        if data is '': continue
        if date[0] not in years:
            years[date[0]] = deepcopy(hours)

        years[date[0]][date[3]][1].append(mk_int(data))
        num_rows += 1

    if num_rows == 0:
        print('dataset empty')
        return
    print('Rendering {} rows'.format(num_rows))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for year, hours in years.items():
        points = []
        for hour, values in hours:
            #plt.scatter(list(range(len(values))), values)

            mean = st.mean(values)
            sd = st.variance(values)**.5

            #print("hour={0}, mean={1:.4f}, s.d.={2:.4f}".format(hour, mean, sd))
            points.append((year, hour, mean))

        x,y,z = list(zip(*points))
        ax.plot(x,y,z, label="{}".format(hour))

    ax.legend()
    ax.set_title('Key: {}'.format(key))
    if show:
        plt.show()

    #plt.title('Mean {} per hour of day. (Error bars equal 1 s.d.)'.format(key))
    #plt.show()
    #draw_2d(dataset)

def draw_3d(dataset, xlabel='x',ylabel='y',zlabel='z'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*list(zip(*dataset)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

def draw_2d(dataset, xlabel='x',ylabel='y'):
    plt.plot(*list(zip(*dataset)))
    plt.show()

def sunrise_by_day():
    global daily

    _daily = sorted(daily, key=lambda x: x[0])
    _items = [(get_linear_date(x[0]), mt2min(x[1]['DAILYSunset'])) for x in _daily]

    draw_2d(_items)

data = [{'key': 'HOURLYVISIBILITY', 'name': 'visibility', 'unit': 'miles'},
        {'key': 'HOURLYRelativeHumidity', 'name': 'Relative Humidity', 'unit': '%'},
        {'key': 'HOURLYWindSpeed', 'name': 'Wind Speed', 'unit': 'mph'},
        {'key': 'HOURLYWindDirection', 'name': 'Wind Direction', 'unit': '°'},
        {'key': 'HOURLYWindGustSpeed', 'name': 'Wind gust speed', 'unit': 'mph'},
        {'key': 'HOURLYStationPressure', 'name': 'Pressure', 'unit': 'inHg'},
        {'key': 'HOURLYPrecip', 'name': 'Hourly Precipitation', 'unit': 'inch to 100th'},
        {'key': 'HOURLYAltimeterSetting', 'name': 'Altimeter setting', 'unit': 'inHg'}]


FILE = '5'
def analyze():
    #sunrise_on_years()
    #wind_dir_vs_date()
    #wind_dir_vs_hour()
    #wind_magnitude_vs_hour(data[9])
    #[wind_magnitude_vs_hour_vs_year(x, show=False) for x in data]
    #[wind_magnitude_vs_year_vs_hour(data[x], show=False) for x in range(len(data))]
    wind_magnitude_vs_day_of_year_per_hour(data[2])
    plt.show()
    #sunrise_by_day()

if __name__ == '__main__':
    print("Using {}".format(FILE))
    #convert()
    read(True, False, False)
    analyze()
    #for i in range(1,6):FILE = str(i); convert();
