#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

####################################################################################
# Data handling 
import pandas as pd
from pandas import concat
from pandas.plotting import scatter_matrix
import numpy as np

####################################################################################
# Visualization
import matplotlib.pyplot as plt
from matplotlib import dates, ticker
from matplotlib.dates import (MONDAY, DateFormatter, MonthLocator, WeekdayLocator, date2num)
import matplotlib as mpl
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.tools import FigureFactory as FF
import plotly.tools as tls

####################################################################################
# TA-Lib: 
import talib

####################################################################################
# Other utilities
import datetime
import time
import os
import sys
import math
from enum import Enum

####################################################################################
####################################################################################
####################################################################################


class MACD_Signal_Listener():
  #-------------------------------------------------
  def __init__(self, _sig_listeners=[]):
    pass
  
  #-------------------------------------------------
  def attachDataFeed(self, df):
    ''' Preloads dataframe from external data feed. 
        @param df: Dataframe. Required columns are OPEN,HIGH,LOW,CLOSE
    '''
    if not 'OPEN' or not 'HIGH' or not 'LOW' or not 'CLOSE' in df.columns:
      printf('ERROR: some column is missing. Required columns are: OPEN,HIGH,LOW,CLOSE')
      return
    self.__df = df[['OPEN','HIGH','LOW','CLOSE']].copy()

  #-------------------------------------------------
  def updateDataFeed(self, df_rows):
    ''' Updates internal dataframe with new data feed entries
        @param df_rows: New data feed entries to inserto into internal dataframe
    '''
    if not 'OPEN' or not 'HIGH' or not 'LOW' or not 'CLOSE' in df_rows.columns:
      printf('ERROR: some column is missing. Required columns are: OPEN,HIGH,LOW,CLOSE')
      return
    self.__df = self.__df.append(df_rows)
  
  #-------------------------------------------------
  def getDataFeed(self):
    return self.__df




