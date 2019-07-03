#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:23:40 2019

@author: tc
"""

import matplotlib.pyplot as plt
import datetime
from matplotlib.widgets import Slider
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY

# (Year, month, day) tuples suffice as args for quotes_historical_yahoo
date1 = (2004, 2, 1)
date2 = (2004, 4, 12)

mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')      # e.g., 12

quotes = quotes_historical_yahoo_ohlc('INTC', date1, date2)
if len(quotes) == 0:
    raise SystemExit

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
#ax.xaxis.set_minor_formatter(dayFormatter)

#plot_day_summary(ax, quotes, ticksize=3)
candlestick_ohlc(ax, quotes, width=0.6)

ax.xaxis_date()
ax.autoscale_view()
plt.axis([datetime.date(*date1).toordinal(), datetime.date(*date1).toordinal()+10, 18.5, 22.5])
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')


axcolor = 'lightgoldenrodyellow'
axpos = plt.axes([0.2, 0.05, 0.65, 0.03], axisbg=axcolor)


spos = Slider(axpos, 'Position', datetime.date(*date1).toordinal(), datetime.date(*date2).toordinal())

def update(val):
    pos = spos.val
    ax.axis([pos,pos+10, 18.5, 22.5])
    fig.canvas.draw_idle()

spos.on_changed(update)

plt.show()