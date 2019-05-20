#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################################
# Librerías de manejo de datos 
import pandas as pd
import numpy as np

####################################################################################
# Librerías de visualización
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.tools import FigureFactory as FF
import plotly.tools as tls

####################################################################################
# TA-Lib: instalación y carga de la librería
import talib

####################################################################################
# Otras utilidades
import datetime
import time
import os
import sys
import math
import pickle
from enum import Enum


####################################################################################
####################################################################################
####################################################################################


class ZIGZAG_Events():
  def __init__(self):
    self.clear()

  def clear(self):
    self.ZIGZAG_Min = False
    self.ZIGZAG_Max = False

  def any(self):
    if self.ZIGZAG_Min or self.ZIGZAG_Max:
      return True
    return False

  def info(self):
    result =''
    if self.ZIGZAG_Min:
      result += 'ZigzagMin '
    if self.ZIGZAG_Max:
      result += 'ZigzagMax '
    return result


####################################################################################
####################################################################################
####################################################################################

class ZIGZAG_Signal_Listener():
  def __init__(self):
    self.__df = None
    self.__events = ZIGZAG_Events()

  def ZIGZAG(self, df, minbars=5, nan_value = np.nan, verbose=False):    
    class ActionCtrl():
      class ActionType(Enum):
        NoActions = 0
        SearchingHigh = 1
        SearchingLow = 2
      def __init__(self, high, low, idx, delta, verbose=False):
        self.curr = ActionCtrl.ActionType.NoActions
        self.last_high = high
        self.last_high_idx = idx
        self.last_low = low
        self.last_low_idx = idx
        self.delta = pd.Timedelta(value=delta)
        self.x = []
        self.y = []
        self.verbose = verbose
        if self.verbose:
          print('New action at idx={}: last_high={}, last_low={}, min-delta={}'.format(idx, self.last_high, self.last_low, self.delta))

      def config(self, min_bars=1, min_percentage=0.0):
        self.min_bars = min_bars
        self.min_percentage = min_percentage

      def __result(self):
        if self.curr == ActionCtrl.ActionType.SearchingHigh:
          return 'high'
        elif self.curr == ActionCtrl.ActionType.SearchingLow:
          return 'low'
        return 'no-action'

      # this function updates HIGH values or LOW values with last recorded depending on the current action
      def zigzag(self, x, df):
        log = 'Procesing row={}'.format(x.name)

        # check if HIGH must be updated
        if self.curr == ActionCtrl.ActionType.SearchingHigh and x.HIGH > self.last_high:
          self.last_high = x.HIGH
          self.last_high_idx = x.name
          log += ' new HIGH={}'.format(x.HIGH)          
          if self.verbose:
            print(log)
          return self.__result()

        # check if LOW must be updated
        if self.curr == ActionCtrl.ActionType.SearchingLow and x.LOW < self.last_low:
          self.last_low = x.LOW
          self.last_low_idx = x.name
          log += ' new LOW={}'.format(x.LOW)
          if self.verbose:
            print(log)
          return self.__result()

        # check if search HIGH starts
        if self.curr != ActionCtrl.ActionType.SearchingHigh and x.HIGH > x.BOLLINGER_HI:
          # check delta condition
          curr_delta = pd.Timedelta(x.name - self.last_low_idx)
          if curr_delta < self.delta:
            log += ' search-high DISCARD. curr-delta={}'.format(curr_delta) 
            if self.verbose:
              print(log)
            return self.__result()
          # save last low     
          df.at[self.last_low_idx,'ZIGZAG'] =  self.last_low   
          self.x.append(self.last_low_idx)
          self.y.append(self.last_low)
          # starts high recording
          self.curr = ActionCtrl.ActionType.SearchingHigh
          self.last_high = x.HIGH
          self.last_high_idx = x.name
          log += ' save ZIGZAG={} at={}, fist HIGH={}'.format(self.last_low, self.last_low_idx, x.HIGH)    
          if self.verbose:
            print(log)
          return self.__result()

        # check if search LOW starts
        if self.curr != ActionCtrl.ActionType.SearchingLow and x.LOW < x.BOLLINGER_LO:
          # check delta condition
          curr_delta = pd.Timedelta(x.name - self.last_high_idx)
          if curr_delta < self.delta:
            log += ' search-low DISCARD. curr-delta={}'.format(curr_delta) 
            if self.verbose:
              print(log)
            return self.__result()
          # save last high
          df.at[self.last_high_idx,'ZIGZAG'] =  self.last_high
          self.x.append(self.last_high_idx)
          self.y.append(self.last_high)
          # starts low recording
          self.curr = ActionCtrl.ActionType.SearchingLow
          self.last_low = x.LOW
          self.last_low_idx = x.name
          log += ' save ZIGZAG={} at={}, first LOW={}'.format(self.last_high, self.last_high_idx, x.LOW)        
          if self.verbose:
            print(log)
          return self.__result()

        if self.curr == ActionCtrl.ActionType.SearchingLow:
          log += ' curr LOW={}'.format(self.last_low)
        elif self.curr == ActionCtrl.ActionType.SearchingHigh:
          log += ' curr HIGH={}'.format(self.last_high)
        if self.verbose:
          print(log)
        return self.__result()    

    # clear events
    self.__events.clear()

    # copy dataframe and calculate bollinger bands if not yet present
    _df = df.copy()
    _df['BOLLINGER_HI'], _df['BOLLINGER_MA'], _df['BOLLINGER_LO'] = talib.BBANDS(_df.CLOSE, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    _df['BOLLINGER_WIDTH'] = _df['BOLLINGER_HI'] - _df['BOLLINGER_LO']
    boll_b = (_df.CLOSE - _df['BOLLINGER_LO'])/(_df['BOLLINGER_HI'] - _df['BOLLINGER_LO'])
    boll_b[np.isnan(boll_b)]=0.5
    boll_b[np.isinf(boll_b)]=0.5
    _df['BOLLINGER_b'] = boll_b
    _df.dropna(inplace=True)
    _df.reset_index(drop=True, inplace=True)

    # Initially no actions are in progress, record first high and low values creating an ActionCtrl object
    action = ActionCtrl(
              high= _df['HIGH'][0], 
              low = _df['LOW'][0], 
              idx = _df.iloc[0].name, 
              delta= minbars*(_df.iloc[1].name - _df.iloc[0].name))

    _df['ZIGZAG'] = nan_value
    _df['ACTION'] = 'no-action'
    _df['ACTION'] = _df.apply(lambda x: action.zigzag(x, _df), axis=1)

    # fills last element as pending
    if _df.ZIGZAG.iloc[-1] == nan_value:
      _df.at[_df.index[-1],'ZIGZAG'] =  action.last_high if _df.ACTION.iloc[-1] == 'high' else action.last_low
      _df.at[_df.index[-1],'ACTION'] =  _df.ACTION.iloc[-1] + '-in-progress'
    
    self.__df = _df
    self.__action = action
    return self.__df[['BOLLINGER_HI', 'BOLLINGER_MA', 'BOLLINGER_LO', 'ZIGZAG', 'ACTION']], self.__action.x, self.__action.y

  def getDataFrame(self):
    return self.__df