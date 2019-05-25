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


class DivergenceEvent():
  def __init__(self, evt, at):
    self.clear()
    if evt == 'regular-bullish-divergence':
      self.RegularBullish = True
    if evt == 'regular-bearish-divergence':
      self.RegularBearish = True
    if evt == 'hidden-bullish-divergence':
      self.HiddenBullish = True
    if evt == 'hidden-bearish-divergence':
      self.HiddenBearish = True
    if at == 'macd':
      self.AtMACD = True
    if at == 'rsi':
      self.AtRSI = True

  def clear(self):
    self.RegularBullish = False
    self.RegularBearish = False
    self.HiddenBullish = False
    self.HiddenBearish = False
    self.AtMACD = False
    self.AtRSI = False

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
      
    return result


####################################################################################
####################################################################################
####################################################################################

class Divergences():
  def __init__(self, level=logging.WARN):    
    self.__logger = logging.getLogger(__name__)
    self.__logger.setLevel(level)
    self.__df = None
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
    self.__events = []

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

    # add result columns
    _df['DIVERGENCE_MACD'] = 'none'
    _df['DIVERGENCE_MACD_FROM'] = 0
    _df['DIVERGENCE_RSI'] = 'none'
    _df['DIVERGENCE_RSI_FROM'] = 0
    _df['DIVERGENCE_STOCH'] = 'none'
    _df['DIVERGENCE_STOCH_FROM'] = 0
    
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
          
          if exitAt == 'zzpoints-processing':
            log += 'zzpoints={} '.format(zzpoints[-6:])
            logger.debug(log)
            return

          curr_is = 'unknown'
          # get indexes of 2 last min and max points:
          if zzpoints.iloc[-1] > zzpoints.iloc[-2]:
            log += 'last is MAX '
            if exitAt == 'minmax-identify':
              logger.debug(log)
              return
            curr_is = 'max'
            max1_idx = zzpoints.index[-3]
            max0_idx = zzpoints.index[-1]
            min1_idx = zzpoints.index[-4]
            min0_idx = zzpoints.index[-2]
            log += 'max0={}, min0={}, max1={}, min1={} '.format(max0_idx, min0_idx, max1_idx, min1_idx)

          else:
            log += 'last is MIN '
            if exitAt == 'minmax-identify':
              logger.debug(log)
              return         
            curr_is = 'min'   
            max1_idx = zzpoints.index[-4]
            max0_idx = zzpoints.index[-2]
            min1_idx = zzpoints.index[-3]
            min0_idx = zzpoints.index[-1]
            log += 'max0={}, min0={}, max1={}, min1={} '.format(max0_idx, min0_idx, max1_idx, min1_idx)

          if exitAt == 'minmax-resolve':
            logger.debug(log)
            return

          # check if is bullish trend
          if ((zzpoints.iloc[-1] >= zzpoints.iloc[-3] and zzpoints.iloc[-3] >= zzpoints.iloc[-5]) or (zzpoints.iloc[-2] >= zzpoints.iloc[-4] and zzpoints.iloc[-4] >= zzpoints.iloc[-6])):
            log += 'bullish-trend '
            # if last is max check regular divergences
            if curr_is == 'max':
              log += 'check macd: {} and {}'.format(df.MACD_main.iloc[max0_idx], df.MACD_main.iloc[max1_idx])
              if df.MACD_main.iloc[max0_idx] < df.MACD_main.iloc[max1_idx]:
                log += 'regular-bearish-divergence '
                df.at[max0_idx, 'DIVERGENCE_MACD'] = 'regular-bearish-divergence'
                df.at[max0_idx, 'DIVERGENCE_MACD_FROM'] = max1_idx
              log += 'check rsi: {} and {}'.format(df.RSI.iloc[max0_idx], df.RSI.iloc[max1_idx])
              if df.RSI.iloc[max0_idx] < df.RSI.iloc[max1_idx]:
                log += 'regular-bearish-divergence '
                df.at[max0_idx, 'DIVERGENCE_RSI'] = 'regular-bearish-divergence'
                df.at[max0_idx, 'DIVERGENCE_RSI_FROM'] = max1_idx
              
            else:
              log += 'check macd: {} and {}'.format(df.MACD_main.iloc[max0_idx], df.MACD_main.iloc[max1_idx])
              if df.MACD_main.iloc[max0_idx] < df.MACD_main.iloc[max1_idx]:
                log += 'hidden-bearish-divergence '
                df.at[min0_idx, 'DIVERGENCE_MACD'] = 'hidden-bearish-divergence'
                df.at[min0_idx, 'DIVERGENCE_MACD_FROM'] = min1_idx
              log += 'check rsi: {} and {}'.format(df.RSI.iloc[max0_idx], df.RSI.iloc[max1_idx])
              if df.RSI.iloc[max0_idx] < df.RSI.iloc[max1_idx]:
                log += 'hidden-bearish-divergence'
                df.at[min0_idx, 'DIVERGENCE_RSI'] = 'hidden-bearish-divergence'
                df.at[min0_idx, 'DIVERGENCE_RSI_FROM'] = min1_idx

          # check if is bearish trend
          elif ((zzpoints.iloc[-1] <= zzpoints.iloc[-3] and zzpoints.iloc[-3] <= zzpoints.iloc[-5]) or (zzpoints.iloc[-2] <= zzpoints.iloc[-4] and zzpoints.iloc[-4] <= zzpoints.iloc[-6])):
            log += 'bearish-trend '
            # if last is max check regular divergences
            if curr_is == 'max':
              log += 'check macd: {} and {}'.format(df.MACD_main.iloc[max0_idx], df.MACD_main.iloc[max1_idx])
              if df.MACD_main.iloc[max0_idx] > df.MACD_main.iloc[max1_idx]:
                log += 'hidden-bullish-divergence '
                df.at[max0_idx, 'DIVERGENCE_MACD'] = 'hidden-bullish-divergence'
                df.at[max0_idx, 'DIVERGENCE_MACD_FROM'] = max1_idx
              log += 'check rsi: {} and {}'.format(df.RSI.iloc[max0_idx], df.RSI.iloc[max1_idx])
              if df.RSI.iloc[max0_idx] > df.RSI.iloc[max1_idx]:
                log += 'hidden-bullish-divergence '
                df.at[max0_idx, 'DIVERGENCE_RSI'] = 'hidden-bullish-divergence'
                df.at[max0_idx, 'DIVERGENCE_RSI_FROM'] = max1_idx
              
            else:
              log += 'check macd: {} and {}'.format(df.MACD_main.iloc[max0_idx], df.MACD_main.iloc[max1_idx])
              if df.MACD_main.iloc[max0_idx] > df.MACD_main.iloc[max1_idx]:
                log += 'regular-bullish-divergence '
                df.at[max0_idx, 'DIVERGENCE_MACD'] = 'regular-bullish-divergence'
                df.at[min0_idx, 'DIVERGENCE_MACD_FROM'] = min1_idx
              log += 'check rsi: {} and {}'.format(df.RSI.iloc[max0_idx], df.RSI.iloc[max1_idx])
              if df.RSI.iloc[max0_idx] > df.RSI.iloc[max1_idx]:
                log += 'regular-bullish-divergence'
                df.at[max0_idx, 'DIVERGENCE_RSI'] = 'regular-bullish-divergence'
                df.at[min0_idx, 'DIVERGENCE_RSI_FROM'] = min1_idx

          # is an undefined trend, then discard calculation
          else: 
            log += 'error-no-trend'
      logger.debug(log)
      #---end-of-search-function

    # execute search
    _df.apply(lambda x: search(x, _df, _nan_value, self.__logger, exitAt), axis=1)
    self.__df = _df

    # check signals on last row
    if self.__df.DIVERGENCE_MACD.iloc[-1] != 'none':
      self.__events.append(DivergenceEvent(self.__df.DIVERGENCE_MACD.iloc[-1], 'macd'))
    if self.__df.DIVERGENCE_RSI.iloc[-1] != 'none':
      self.__events.append(DivergenceEvent(self.__df.DIVERGENCE_RSI.iloc[-1], 'rsi'))
    return self.__df, self.__events

