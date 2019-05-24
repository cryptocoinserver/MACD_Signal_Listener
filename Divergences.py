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
from ZIGZAG_Signal_Listener import ZIGZAG_Signal_Listener

####################################################################################
# Otras utilidades
import datetime
import time
import os
import sys
import math
import pickle
from enum import Enum
import logging


####################################################################################
####################################################################################
####################################################################################


class DivergenceEvents():
  def __init__(self):
    self.clear()

  def clear(self):
    self.RegularBullish = False
    self.RegularBearish = False
    self.HiddenBullish = False
    self.HiddenBearish = False
    self.AtMACD = False
    self.AtRSI = False
    self.AtStochastic = False

  def any(self):
    if self.RegularBullish or self.RegularBearish or self.HiddenBullish or self.HiddenBearish:
      return True
    return False

  def info(self):
    result =''
    if self.RegularBullish:
      result += 'RegularBullish '
    if self.RegularBearish:
      result += 'RegularBearish '
    if self.HiddenBullish:
      result += 'HiddenBullish '
    if self.HiddenBearish:
      result += 'HiddenBearish '
    if len(result) > 0:
      result += 'at '
    if self.AtMACD:
      result += 'AtMACD '
    if self.AtRSI:
      result += 'AtRSI '
    if self.AtStochastic:
      result += 'AtStochastic '
      
    return result


####################################################################################
####################################################################################
####################################################################################

class Divergences():
  def __init__(self, level=logging.WARN):    
    self.__logger = logging.getLogger(__name__)
    self.__logger.setLevel(level)
    self.__df = None
    self.__events = DivergenceEvents()
    self.__logger.info('Created!')
    self.__zigzag = ZIGZAG_Signal_Listener(level)

  def searchDivergences(self, df, zigzag_cfg = dict(), level=logging.WARN):    
    """Builds from dataframe df, next indicators: MACD, RSI, Stochastic with default
       parameters. Then builds a zigzag indicator.

       Returns resulting dataframe and events raised at last bar.

    Keyword arguments:
      df -- Datafeed to apply indicator
      minbars -- Min. number of bars per flip to avoid discarding (default 12)
      zigzag_cfg -- Dictionary with zigzag configuration (default None)
      level -- logging level (default WARN)
    """
    
    # clear events
    self.__events.clear()

    #builds zigzag
    _minbars   = zigzag_cfg['minbars'] if 'minbars' in zigzag_cfg.keys() else 12 , 
    _bb_period = zigzag_cfg['bb_period'] if 'bb_period' in zigzag_cfg.keys() else 20, 
    _bb_dev    = zigzag_cfg['bb_dev'] if 'bb_dev' in zigzag_cfg.keys() else 2.0,
    _bb_sma    = zigzag_cfg['bb_sma'] if 'bb_sma' in zigzag_cfg.keys() else [100],
    _nan_value = zigzag_cfg['nan_value'] if 'nan_value' in zigzag_cfg.keys() else 0.0, 
    _df, _x, _y, _evt_zigzag =  self.__zigzag.ZIGZAG( df, 
                                                      minbars   = _minbars,
                                                      bb_period = _bb_period,
                                                      bb_dev    = _bb_dev,
                                                      bb_sma    = _bb_sma,
                                                      nan_value = _nan_value
                                                      level     = level)

    # build MACD_main, RSI and Stochastic
    _df['MACD_main'], _df['MACD_signal'], _df['MACD_hist'] = talib.MACD(_df.CLOSE, fastperiod=12, slowperiod=26, signalperiod=9)
    _df['RSI'] = talib.RSI(_df.CLOSE, timeperiod=14)
    _df['STOCH_K'], _df['STOCH_d'] = talib.STOCH(_df.HIGH, _df.LOW, _df.CLOSE, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # remove Nans and reindex from sample 0
    _df.dropna(inplace=True)
    _df.reset_index(drop=True, inplace=True)

    # executes divergence localization process:
    # 1. Set a default trend: requires 3 max and 3 min points
    # 1a.If max increasing or min increasing -> Bullish trend
    # 1b.If max decreasing or min decreasing -> Bearish trend
    # 1c.Else discard.

    def search(row, df, nan_value):
      # skip rows where no zzpoints
      if row.ZIGZAG == nan_value: return
      # get last 6 zigzag points
      zzpoints = df.ZIGZAG[(df.index <= row.index) & (df.ZIGZAG != nan_value)]
      # discard if no 6 points
      if zzpoints.shape[0] < 6: return
      # get indexes of 2 last min and max points:
      if zzpoints[-1] > zzpoints[-2]:
        max1_idx = zzpoints[-3].index
        max0_idx = zzpoints[-1].index
        min1_idx = zzpoints[-4].index
        min0_idx = zzpoints[-2].index
      else:
        # get indexes of last 2 min and max points
        max1_idx = zzpoints[-4].index
        max0_idx = zzpoints[-2].index
        min1_idx = zzpoints[-3].index
        min0_idx = zzpoints[-1].index

      # check if is bullish trend
      if ((zzpoints[-1] >= zzpoints[-3] and zzpoints[-3] >= zzpoints[-5]) or (zzpoints[-2] >= zzpoints[-4] and zzpoints[-4] >= zzpoints[-6])):

      # check if is bearish trend
      elif ((zzpoints[-1] <= zzpoints[-3] and zzpoints[-3] <= zzpoints[-5]) or (zzpoints[-2] <= zzpoints[-4] and zzpoints[-4] <= zzpoints[-6])):
      
      # is an undefined trend, then discard calculation
      else: return 
      #---end-of-search-function

    _df.apply(lambda x: self.__search(x, _df, _nan_value), axis=1)

    # 1. Load first 2 consecutive max in zigzag indicator
    # 2. If increasing then check if regular bearish divergence on any oscillator
    # 1b. If decreasing then check if hidden bullish divergence on any oscillator

    # Initially no actions are in progress, record first high and low values creating an ActionCtrl object
    action = ActionCtrl(
              high= _df['HIGH'][0], #max(_df['OPEN'][0], _df['CLOSE'][0]), 
              low = _df['LOW'][0], #min(_df['OPEN'][0], _df['CLOSE'][0]), 
              idx = _df.iloc[0].name, 
              delta= minbars,
              level=level)

    _df['ZIGZAG'] = nan_value
    _df['ACTION'] = 'no-action'
    _df['ACTION'] = _df.apply(lambda x: action.zigzag(x, _df), axis=1)

    # fills last element as pending
    if _df.ZIGZAG.iloc[-1] == nan_value:
      _df.at[_df.index[-1],'ZIGZAG'] =  action.last_high if _df.ACTION.iloc[-1] == 'high' else action.last_low
      _df.at[_df.index[-1],'ACTION'] =  _df.ACTION.iloc[-1] + '-in-progress'
    
    self.__df = _df
    self.__action = action
    return self.__df, self.__action.x, self.__action.y, self.__action.events

  def getDataFrame(self):
    return self.__df