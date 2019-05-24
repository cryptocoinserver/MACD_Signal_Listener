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

  def searchDivergences(self, df, zigzag_cfg = dict(), level=logging.WARN, exitAt='end'):    
    """Builds from dataframe df, next indicators: MACD, RSI, Stochastic with default
       parameters. Then builds a zigzag indicator.

       Returns resulting dataframe and events raised at last bar.

    Keyword arguments:
      df -- Datafeed to apply indicator
      minbars -- Min. number of bars per flip to avoid discarding (default 12)
      zigzag_cfg -- Dictionary with zigzag configuration (default None)
      level -- logging level (default WARN)
      exitAt -- Label for exit at debugging phase
    """
    
    # clear events
    self.__events.clear()

    #builds zigzag
    _minbars   = zigzag_cfg['minbars'] if 'minbars' in zigzag_cfg.keys() else 12  
    _bb_period = zigzag_cfg['bb_period'] if 'bb_period' in zigzag_cfg.keys() else 2 
    _bb_dev    = zigzag_cfg['bb_dev'] if 'bb_dev' in zigzag_cfg.keys() else 2.0
    _bb_sma    = zigzag_cfg['bb_sma'] if 'bb_sma' in zigzag_cfg.keys() else [100]
    _nan_value = zigzag_cfg['nan_value'] if 'nan_value' in zigzag_cfg.keys() else 0.0 
    _zlevel    = zigzag_cfg['level'] if 'level ' in zigzag_cfg.keys() else logging.WARN
    _df, _x, _y, _evt_zigzag =  self.__zigzag.ZIGZAG( df, 
                                                      minbars   = _minbars,
                                                      bb_period = _bb_period,
                                                      bb_dev    = _bb_dev,
                                                      bb_sma    = _bb_sma,
                                                      nan_value = _nan_value,
                                                      level     = _zlevel)

    if exitAt == 'zigzag-calculation':
      return _df.copy()

    # build MACD_main, RSI and Stochastic
    _df['MACD_main'], _df['MACD_signal'], _df['MACD_hist'] = talib.MACD(_df.CLOSE, fastperiod=12, slowperiod=26, signalperiod=9)
    _df['RSI'] = talib.RSI(_df.CLOSE, timeperiod=14)
    _df['STOCH_K'], _df['STOCH_d'] = talib.STOCH(_df.HIGH, _df.LOW, _df.CLOSE, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # remove Nans and reindex from sample 0
    _df.dropna(inplace=True)
    _df.reset_index(drop=True, inplace=True)

    if exitAt == 'oscillator-built':
      return _df.copy()

    # executes divergence localization process:
    # 1. Set a default trend: requires 3 max and 3 min points
    # 1a.If max increasing or min increasing -> Bullish trend
    # 1b.If max decreasing or min decreasing -> Bearish trend
    # 1c.Else discard.

    def search(row, df, nan_value, logger, exitAt):
      log = 'row [{}]: '.format(row.name)
      if exitAt == 'row-count-processing':
        logger.debug(log)
        return

      # skip rows where no zzpoints
      if row.ZIGZAG == nan_value: 
        log += 'error-zigzag-isnan'

      else:
        # get last 6 zigzag points
        zzpoints = df.ZIGZAG[(df.index <= row.name) & (df.ZIGZAG != nan_value)]
        # discard if no 6 points
        if zzpoints.shape[0] < 6: 
           log += 'error-zzpoints-count={} '.format(zzpoints.shape[0])
        
        else:
          log += 'zzpoints={} '.format(zzpoints[-6:])
          if exitAt == 'zzpoints-processing':
            logger.debug(log)
            return

          # get indexes of 2 last min and max points:
          if zzpoints[-1] > zzpoints[-2]:
            log += 'last is MAX '
            max1_idx = zzpoints.index[-3]
            max0_idx = zzpoints.index[-1]
            min1_idx = zzpoints.index[-4]
            min0_idx = zzpoints.index[-2]
            log += 'max0={}, min0={}, max1={}, min1={} '.format(max0_idx, min0_idx, max1_idx, min1_idx)
            #TODO<------------

          else:
            log += 'last is MIN '
            # get indexes of last 2 min and max points
            max1_idx = zzpoints.index[-4]
            max0_idx = zzpoints.index[-2]
            min1_idx = zzpoints.index[-3]
            min0_idx = zzpoints.index[-1]
            log += 'max0={}, min0={}, max1={}, min1={} '.format(max0_idx, min0_idx, max1_idx, min1_idx)
            #TODO<------------        

          # check if is bullish trend
          if ((zzpoints[-1] >= zzpoints[-3] and zzpoints[-3] >= zzpoints[-5]) or (zzpoints[-2] >= zzpoints[-4] and zzpoints[-4] >= zzpoints[-6])):
            log += 'bullish-trend '
          # check if is bearish trend
          elif ((zzpoints[-1] <= zzpoints[-3] and zzpoints[-3] <= zzpoints[-5]) or (zzpoints[-2] <= zzpoints[-4] and zzpoints[-4] <= zzpoints[-6])):
            log += 'bearish-trend '
          # is an undefined trend, then discard calculation
          else: 
            log += 'error-no-trend'
      logger.debug(log)
      #---end-of-search-function

    # execute search
    self.__df = _df.apply(lambda x: search(x, _df, _nan_value, self.__logger, exitAt), axis=1)
    return self.__df

