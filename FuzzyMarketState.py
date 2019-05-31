#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  FuzzyMarketState is a python class that represents the current market state
  conditions by using Fuzzy Logic semanthics.

  In order to achieve the commented output, it uses several technical indicators
  to build sinthetic variables, that afterwards will fuzzify into more abstract
  definition of the market status.

  Technical indicators
  ====================

  - Zigzag
  - Bollinger
  - MACD
  - RSI
  - Fibonacci
  - Moving averages
  - Supports & Resistances
  - Dynamic channels

  Sinthetic Fuzzy variables
  =========================

  - Proximiy of price to:
    - Relevant SMA
    - Specific SMA
    - Relevant support/resistance
    - Relevant fibo level
    - Specific fibo level
    - Current dynamic resistance
    - Current dynamic support
  
  - Force of Patterns:
    - Divergence (regular|hidden, bullish|bearish)
    - Trend (bullish|bearish)
    - Candlestick (bullish|bearish)
"""

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
import logging



####################################################################################
####################################################################################
####################################################################################

class FuzzyMarketState():
  def __init__(self, level=logging.WARN):    
    self.__logger = logging.getLogger(__name__)
    self.__logger.setLevel(level)
    self.__logger.info('Created!')
    self.__zigzag = ZIGZAG_Signal_Listener(level)

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def setLoggingLevel(self, level): 
    """ Updates loggging level

      Keyword arguments:
        level -- new logging level
    """   
    self.__logger.setLevel(level)
    self.__logger.debug('Logging level changed to {}'.format(level))

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def loadCSV(self, file, sep=';'):    
    """ Loads a csv file imported from the trading platform with this
        columns: DATE,TIME,OPEN,HIGH,LOW,CLOSE,TICKVOL,VOL,SPREAD.
        DATE and TIME are converted to datetime as new column TIME, and
        index is reset to start from sample 0.
      
      Keyword arguments:
        file -- csv file
        sep -- character separator (default ';')
      Returns:
        num_rows -- number of rows loaded        
    """
    _df = pd.read_csv(file, sep)
    _df['TIME'] = _df['DATE'] + '  ' + _df['TIME'] 
    _df['TIME'] = _df['TIME'].map(lambda x: datetime.datetime.strptime(x, '%Y.%m.%d %H:%M:%S'))  
    _df['TIME'] = pd.to_datetime(_df['TIME'])
    self.__df = _df.copy()
    self.__logger.debug('loaded {} rows from {} to {}'.format(self.__df.shape[0], _df['TIME'].iloc[0], _df['TIME'].iloc[-1]))
    return self.__df

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def loadDataframe(self, df):    
    """ Loads a dataframe
      
      Keyword arguments:
        df -- dataframe
    """
    self.__df = df.copy()

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def getDataframe(self):    
    """ gets a reference to the internal dataframe
      
      Returns:
        self.__df -- dataframe
    """
    return self.__df

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildIndicators(self, params = dict(), level=None):    
    """ Builds default integrated indicators for all rows in integrated dataframe, such as: 
        - Zigzag (includes BollingerBands and derivatives)
        - MACD
        - RSI
        - MovingAverages (SMA, EMA)
 
        Also builds Fuzzy indicators based on previous ones, such as:
        - Price vs Overbought level:  [Far, Near, In]
        - Price vs Oversell level: [Far, Near, In]
        - Trend: [HardBearish, SoftBearish, NoTrend, SoftBullish, HardBullish]
        - Divergence: [HardBearish, SoftBearish, NoDivergence, SoftBullish, HardBullish]
        - Fibo retracements 23%, 38%, 50%, 61%: [Far, Near, In]
        - Fibo extensions_1 123%, 138%, 150%, 161%: [Far, Near, In]
        - Fibo extensions_2 223%, 238%, 250%, 261%: [Far, Near, In]
        - MovingAverage proximity: [Far, Near, In]
        - SupportResistance proximity: [Far, Near, In]
        - DynamicTrendLine proximity: [Far, Near, In]
        - Candlestick patterns: [NotPresent, Soft, Hard]
      
      Keyword arguments:
        params -- dictionary with configuration parameters for indicators (default None), such us:

      Return:
        self.__df -- Internal dataframe with all indicators
     """
    prev_level = self.__logger.level
    if level is not None:
      self.__logger.setLevel(level)

    # build zigzag indicator (includes BollingerBands and derivatives)   
    minbars   = params['zz_minbars'] if 'zz_minbars' in params.keys() else 12  
    bb_period = params['bb_period'] if 'bb_period' in params.keys() else 2 
    bb_dev    = params['bb_dev'] if 'bb_dev' in params.keys() else 2.0
    bb_sma    = params['bb_sma'] if 'bb_sma' in params.keys() else [100]
    nan_value = params['zz_nan_value'] if 'zz_nan_value' in params.keys() else 0.0 
    _df = self.buildZigzag(self.__df, minbars, bb_period, bb_dev, bb_sma, nan_value)

    # build oscillators (includes MACD and RSI)
    macd_applied  = params['macd_applied'] if 'macd_applied' in params.keys() else 'CLOSE'
    macd_fast     = params['macd_fast'] if 'macd_fast' in params.keys() else 12 
    macd_slow     = params['macd_slow'] if 'macd_slow' in params.keys() else 26
    macd_sig      = params['macd_sig'] if 'macd_sig' in params.keys() else 9  
    rsi_applied   = params['rsi_applied'] if 'rsi_applied' in params.keys() else 'CLOSE'
    rsi_period    = params['rsi_period'] if 'rsi_period' in params.keys() else 14     
    self.buildOscillators(_df, macd_applied, macd_fast, macd_slow, macd_sig, rsi_applied, rsi_period)

    # build 3 moving averages (includes SMA50, SMA100, SMA200)
    ma_fast_applied = params['ma_fast_applied'] if 'ma_fast_applied' in params.keys() else 'CLOSE'
    ma_fast_period  = params['ma_fast_period'] if 'ma_fast_period' in params.keys() else 50 
    ma_fast_type    = params['ma_fast_type'] if 'ma_fast_type' in params.keys() else 'SMA'
    ma_mid_applied  = params['ma_mid_applied'] if 'ma_mid_applied' in params.keys() else 'CLOSE'
    ma_mid_period   = params['ma_mid_period'] if 'ma_mid_period' in params.keys() else 100 
    ma_mid_type     = params['ma_mid_type'] if 'ma_mid_type' in params.keys() else 'SMA'
    ma_slow_applied = params['ma_slow_applied'] if 'ma_slow_applied' in params.keys() else 'CLOSE'
    ma_slow_period  = params['ma_slow_period'] if 'ma_slow_period' in params.keys() else 200 
    ma_slow_type    = params['ma_slow_type'] if 'ma_slow_type' in params.keys() else 'SMA'
    self.build3MovingAverages(_df, 
                              ma_fast_applied, ma_fast_period, ma_fast_type,
                              ma_mid_applied, ma_mid_period, ma_mid_type,
                              ma_slow_applied, ma_slow_period, ma_slow_type)

    # build fibonacci retracement and extensions
    nan_value = params['fibo_nan_value'] if 'fibo_nan_value' in params.keys() else 0.0
    fibo_level = params['fibo_level'] if 'fibo_level' in params.keys() else nan_value 
    self.buildCommonFiboLevels(_df, nan_value)    

    # build support and resistances based on previous zigzags
    nan_value = params['sr_nan_value'] if 'sr_nan_value' in params.keys() else 0.0 
    self.buildSupports(_df, nan_value)
    self.buildResistances(_df, nan_value)

    # build dynamic support-resistance of channel based on previous zigzags
    nan_value = params['channel_nan_value'] if 'channel_nan_value' in params.keys() else 0.0 
    self.buildChannel(_df, nan_value)    

    # build trend detector based on different indicators
    nan_value = params['trend_nan_value'] if 'trend_nan_value' in params.keys() else 0.0
    fibo_level = params['trend_fibo_level'] if 'trend_fibo_level' in params.keys() else 0.04 
    self.buildTrends(_df, nan_value)

    # build divergence detector based on zigzag, macd and rsi
    self.buildDivergences(_df, nan_value)
    
    # remove NaN values and reindex from sample 0
    _df.dropna(inplace=True)
    _df.reset_index(drop=True, inplace=True)

    # restore logging level
    if level is not None:
      self.__logger.setLevel(prev_level)

    # updates and return current dataframe
    self.__df = _df
    return self.__df
  

  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildZigzag(self, df, minbars, bb_period, bb_dev, bb_sma, nan_value):
    """ Builds zigzag indicator

      Keyword arguments:
        df -- dataframe
        zz_minbars -- Min. number of bars per flip to avoid discarding (default 12)
        bb_period -- Bollinger bands period (default 20)
        bb_dev -- Bollinger bands deviation (default 2.0)
        bb_sma -- List of SMA timeperiod for Bands Width SMA calculation
        nan_value -- Values for zigzag indicator during search phase (default 0.0)
        zlevel -- logging level (default WARN)
      Return:
        _df -- Copy dataframe from df with zigzag columns
    """

    _df, _=  self.__zigzag.ZIGZAG(df, 
                                  minbars   = minbars,
                                  bb_period = bb_period,
                                  bb_dev    = bb_dev,
                                  bb_sma    = bb_sma,
                                  nan_value = nan_value,
                                  level     = self.__logger.level)
    # add columns for trend detection using zigzag
    _df['ZZ_BULLISH_TREND'] = _df.apply(lambda x: 1 if x.P1 > x.P3 and x.P3 > x.P5 and x.P2 > x.P4 else 0, axis= 1)
    _df['ZZ_BEARISH_TREND'] = _df.apply(lambda x: 1 if x.P1 < x.P3 and x.P3 < x.P5 and x.P2 < x.P4 else 0, axis= 1)
    return _df

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildOscillators(self, df, macd_applied, macd_fast, macd_slow, macd_sig, rsi_applied, rsi_period):
    """ Builds different oscillators like MACD and RSI

      Keyword arguments:
        df -- dataframe
        macd_applied -- Price to apply MACD (default 'CLOSE')
        macd_fast -- MACD fast period (default 12)
        macd_slow -- MACD slow period (default 26)
        macd_sig -- MACD signal period (default 9)
        rsi_applied -- Price to apply RSI (default 'CLOSE')
        rsi_period -- RSI period (default 14)
      Return:
        dict -- Dictionary with Macd and RSI common series (main, signals, ...)
    """
    _macd_main, _macd_sig, _macd_hist = talib.MACD(df[macd_applied], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_sig)
    # add crossovers between main and zero level
    _macd_cross_zero_up = ((_macd_main > 0) & (_macd_main.shift(1) < 0 | ((_macd_main.shift(1)==0) & (_macd_main.shift(2) < 0))))
    _macd_cross_zero_dn = ((_macd_main < 0) & (_macd_main.shift(1) > 0 | ((_macd_main.shift(1)==0) & (_macd_main.shift(2) > 0))))
    # add crossovers between main and signal lines
    _macd_cross_sig_up = (((_macd_main > _macd_sig) & ((_macd_main.shift(1) < _macd_sig.shift(1)) | ((_macd_main.shift(1)==_macd_sig.shift(1)) & (_macd_main.shift(2) < _macd_sig.shift(2))))) & ((_macd_main < 0) & (_macd_main.shift(1) < 0) & (_macd_main.shift(2) < 0)))
    _macd_cross_sig_dn = (((_macd_main < _macd_sig) & ((_macd_main.shift(1) > _macd_sig.shift(1)) | ((_macd_main.shift(1)==_macd_sig.shift(1)) & (_macd_main.shift(2) > _macd_sig.shift(2))))) & ((_macd_main > 0) & (_macd_main.shift(1) > 0) & (_macd_main.shift(2) > 0)))


    _rsi = talib.RSI(df[rsi_applied], timeperiod=rsi_period)
    # add crossovers between overbought levels
    _rsi_cross_ob_up = ((_rsi > 70) & (_rsi.shift(1) < 70 | ((_rsi.shift(1)==70) & (_rsi.shift(2) < 70))))
    _rsi_cross_ob_dn = ((_rsi < 70) & (_rsi.shift(1) > 70 | ((_rsi.shift(1)==70) & (_rsi.shift(2) > 70))))
    # add crossovers between oversell levels
    _rsi_cross_os_up = ((_rsi > 30) & (_rsi.shift(1) < 30 | ((_rsi.shift(1)==30) & (_rsi.shift(2) < 30))))
    _rsi_cross_os_dn = ((_rsi < 30) & (_rsi.shift(1) > 30 | ((_rsi.shift(1)==30) & (_rsi.shift(2) > 30))))

    df['MACD_main']           = _macd_main
    df['MACD_sig']            = _macd_sig
    df['MACD_hist']           = _macd_hist
    df['MACD_CROSS_ZERO_UP']  = _macd_cross_zero_up
    df['MACD_CROSS_ZERO_DN']  = _macd_cross_zero_dn
    df['MACD_CROSS_SIG_UP']   = _macd_cross_sig_up
    df['MACD_CROSS_SIG_DN']   = _macd_cross_sig_dn
    df['RSI']                = _rsi
    df['RSI_cross_ob_up']    = _rsi_cross_ob_up
    df['RSI_cross_ob_dn']    = _rsi_cross_ob_dn
    df['RSI_cross_os_up']    = _rsi_cross_os_up
    df['RSI_cross_os_dn']    = _rsi_cross_os_dn

    return {'macd': {
              'main': _macd_main, 
              'sig': _macd_sig, 
              'hist': _macd_hist,
              'cross_zero_up' : _macd_cross_zero_up,
              'cross_zero_dn' : _macd_cross_zero_dn,
              'cross_sig_up' : _macd_cross_sig_up,
              'cross_sig_dn' : _macd_cross_sig_dn
            }, 
            'rsi':{
              'main': _rsi,
              'cross_ob_up' : _rsi_cross_ob_up,
              'cross_ob_dn' : _rsi_cross_ob_dn,
              'cross_os_up' : _rsi_cross_os_up,
              'cross_os_dn' : _rsi_cross_os_dn
            }
            }

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def build3MovingAverages( self, df, 
                              ma_fast_applied, ma_fast_period, ma_fast_type, 
                              ma_mid_applied, ma_mid_period, ma_mid_type, 
                              ma_slow_applied, ma_slow_period, ma_slow_type):
    """ Builds different moving averages according with the type and periods

      Keyword arguments:
        df -- dataframe
        ma_x_applied: Price to apply MA
        ma_x_type: MA type 
        ma_x_period: MA period
      Return:
        _ma_list -- List of SMA 
    """
    if ma_fast_type == 'EMA':
      ma_fast = talib.EMA(df[ma_fast_applied], timeperiod=ma_fast_period)
    else:
      ma_fast = talib.SMA(df[ma_fast_applied], timeperiod=ma_fast_period)
    if ma_mid_type == 'EMA':
      ma_mid = talib.EMA(df[ma_mid_applied], timeperiod=ma_mid_period)
    else:
      ma_mid = talib.SMA(df[ma_mid_applied], timeperiod=ma_mid_period)
    if ma_slow_type == 'EMA':
      ma_slow = talib.EMA(df[ma_slow_applied], timeperiod=ma_slow_period)
    else:
      ma_slow = talib.SMA(df[ma_slow_applied], timeperiod=ma_slow_period)
    
    df['SMA_FAST']  = ma_fast
    df['SMA_MID']   = ma_mid
    df['SMA_SLOW']  = ma_slow
    df['SMA_BULLISH_TREND'] = df.apply(lambda x: 1 if x.SMA_FAST > x.SMA_MID and x.SMA_MID > x.SMA_SLOW else 0, axis=1)
    df['SMA_BEARISH_TREND'] = df.apply(lambda x: 1 if x.SMA_FAST < x.SMA_MID and x.SMA_MID < x.SMA_SLOW else 0, axis=1)

    return {'sma_fast': ma_fast, 
            'sma_mid': ma_mid, 
            'sma_slow': ma_slow,
            'sma_bullish_trend': df['SMA_BULLISH_TREND'],
            'sma_bearish_trend': df['SMA_BEARISH_TREND']}

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildCommonFiboLevels(self, df, nan_value):
    self.buildFiboLevel(df, 'FIBO_CURR', nan_value)
    self.buildFiboLevel(df, 'FIBO_023', nan_value, fibo_level=0.236)
    self.buildFiboLevel(df, 'FIBO_038', nan_value, fibo_level=0.382)
    self.buildFiboLevel(df, 'FIBO_050', nan_value, fibo_level=0.500)
    self.buildFiboLevel(df, 'FIBO_061', nan_value, fibo_level=0.618)
    self.buildFiboLevel(df, 'FIBO_078', nan_value, fibo_level=0.786)
    self.buildFiboLevel(df, 'FIBO_123', nan_value, fibo_level=1.236)
    self.buildFiboLevel(df, 'FIBO_138', nan_value, fibo_level=1.382)
    self.buildFiboLevel(df, 'FIBO_150', nan_value, fibo_level=1.500)
    self.buildFiboLevel(df, 'FIBO_161', nan_value, fibo_level=1.618)
    self.buildFiboLevel(df, 'FIBO_178', nan_value, fibo_level=1.786)

  def buildFiboLevel(self, df, name, nan_value, fibo_level=0.0):
    """ Builds fibo level depending on zigzag points

      Keyword arguments:
        df -- dataframe
        name -- name of the column to build
        nan_value -- NaN value for empty results
        fibo_level -- Fibo level to calculate (default 0.0)
      Return:
        fibo -- Fibo level 
    """
    def fibo_retr(x, df, nan_value, fibo_level):
      value = x.ZIGZAG if x.ZIGZAG != nan_value else x.HIGH if x.P1 < x.P2 else x.LOW
      if x.P1 > x.P2:
        if fibo_level == 0.0:
          return (x.P1 - value)/(x.P1 - x.P2)
        else:
          return (x.P1 - ((x.P1 - x.P2)*fibo_level))
      else:
        if fibo_level == 0.0:
          return (value - x.P1)/(x.P2 - x.P1)
        else:
          return (x.P1 + ((x.P2 - x.P1)*fibo_level))
   
    fibo = df.apply(lambda x: fibo_retr(x, df, nan_value, fibo_level), axis=1)  
    df[name] = fibo
    return {'fibo': fibo} 

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildSupports(self, df, nan_value):
    """ Builds 2 main supports based on current zigzag trend points

      Keyword arguments:
        df -- dataframe
        nan_value -- NaN value for empty results
      Return:
        support1,support2 -- Two main support levels
    """    
    def fn_support(x, df, level, nan_value):
      if x.ZZ_BULLISH_TREND == 1:
        if x.P1 > x.P2:
          if level == 1:
            return x.P2
          else:
            return x.P4
        else:
          if level == 1:
            return x.P1
          else:
            return x.P2
      return nan_value
     
    df['SUPPORT_1'] = df.apply(lambda x: fn_support(x, df, 1, nan_value), axis=1)  
    df['SUPPORT_2'] = df.apply(lambda x: fn_support(x, df, 2, nan_value), axis=1)  
    return {'support_1': df['SUPPORT_1'], 'support_2': df['SUPPORT_2']}    

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildResistances(self, df, nan_value):
    """ Builds 2 main resistances based on current zigzag trend points

      Keyword arguments:
        df -- dataframe
        nan_value -- NaN value for empty results
      Return:
        resist1,resist2 -- Two main resistance levels
    """
    def fn_resistance(x, df, level, nan_value):
      if x.ZZ_BEARISH_TREND == 1:
        if x.P1 > x.P2:
          if level == 1:
            return x.P1
          else:
            return x.P2
        else:
          if level == 1:
            return x.P2
          else:
            return x.P4
      return nan_value
     
    df['RESISTANCE_1'] = df.apply(lambda x: fn_resistance(x, df, 1, nan_value), axis=1)  
    df['RESISTANCE_2'] = df.apply(lambda x: fn_resistance(x, df, 2, nan_value), axis=1)  
    return {'resistance_1': df['RESISTANCE_1'], 'resistance_2': df['RESISTANCE_2']}   

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildChannel(self, df, nan_value):
    """ Builds 2 main channel limits based on current zigzag trend points

      Keyword arguments:
        df -- dataframe
        nan_value -- NaN value for empty results
      Return:
        ch_up, ch_dwon -- Two main channel lines
    """    
    def fn_channel(x, df, level, nan_value):
      def line(x0,x1,y0,y1,x):
        return (((y1-y0)/(x1-x0))*(x-x0))+y0

      if x.ZZ_BULLISH_TREND == 1 or x.ZZ_BEARISH_TREND == 1:
        if x.P1 > x.P2:
          if level == 1:
            return line(x.P3_idx, x.P1_idx, x.P3, x.P1, x.name)
          else:
            return line(x.P4_idx, x.P2_idx, x.P4, x.P2, x.name)
        else:
          if level == 1:
            return line(x.P4_idx, x.P2_idx, x.P4, x.P2, x.name)
          else:
            return line(x.P3_idx, x.P1_idx, x.P3, x.P1, x.name)
      return nan_value      
      
    df['CHANNEL_UPPER_LIMIT'] = df.apply(lambda x: fn_channel(x, df, 1, nan_value), axis=1)  
    df['CHANNEL_LOWER_LIMIT'] = df.apply(lambda x: fn_channel(x, df, 2, nan_value), axis=1)  
    return {'channel_upper_limit': df['CHANNEL_UPPER_LIMIT'], 'channel_lower_limit': df['CHANNEL_LOWER_LIMIT']}   

  
  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildTrends(self, df, nan_value):
    """ Builds a trend detector based on different indicators, as follows:
        ZIGZAG_TREND_DETECTOR: provides a trend feedback according with its last
        zigzag points.
        SMA_TREND_DETECTOR: provides trend feedback according with its fast-mid-slow
        moving averages
        FIBO_TREND_DETECTOR: provides trend feedback according with fibo retracements
        and extensions.

        Trend will have different strength [0..1] depending on which indicators are satisfied:
        strength = 0.0 => None indicators are satisfied
        strength = 1.0 => All indicators are satisfied
        strenght + 0.5 => SMA_TREND is satisfied
        strenght + 0.3 => ZIGZAG_TREND is satisfied
        strenght + 0.2 => FIBO_TREND is satisfied

      Keyword arguments:
        df -- dataframe
        nan_value -- NaN value for empty results
        fibo_tol -- Tolerance % around fibo levels
      Return:
        up_trend, down_trend -- Series containing strength of each trend
    """    
    def fn_trend(x, df, level, nan_value, fibo_tol):
      strength = 0.0
      if level == 1:
        if x.SMA_BULLISH_TREND == 1:
          strength += 0.5
        if x.ZZ_BULLISH_TREND == 1:
          strength += 0.3
        if x.P1 > x.P2 and x.FIBO_CURR > (0.236-fibo_tol) and (x.FIBO_CURR < 0.618+fibo_tol):
          strength += 0.15
        if x.P1 < x.P2 and x.FIBO_CURR > (1.236-fibo_tol) and (x.FIBO_CURR < 1.618+fibo_tol):
          strength += 0.05
      else:
        if x.SMA_BEARISH_TREND == 1:
          strength += 0.5
        if x.ZZ_BEARISH_TREND == 1:
          strength += 0.3
        if x.P1 < x.P2 and x.FIBO_CURR > (0.236-fibo_tol) and (x.FIBO_CURR < 0.618+fibo_tol):
          strength += 0.15
        if x.P1 > x.P2 and x.FIBO_CURR > (1.236-fibo_tol) and (x.FIBO_CURR < 1.618+fibo_tol):
          strength += 0.05
      return strength
     
    df['BULLISH_TREND'] = df.apply(lambda x: fn_trend(x, df, 1, nan_value, fibo_level), axis=1)  
    df['BEARISH_TREND'] = df.apply(lambda x: fn_trend(x, df, 2, nan_value, fibo_level), axis=1)  
    return {'bullish_trend': df['BULLISH_TREND'], 'bearish_trend': df['BEARISH_TREND']}   


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def buildDivergences(self, df, nan_value):    
    """Builds divergences based on zigzag, macd and rsi

      Keyword arguments:
        df -- dataframe
        nan_value -- NaN value for empty results
        fibo_tol -- Tolerance % around fibo levels
      Return:
        regbulldiv, hidbulldiv, regbeardiv, hidbeardiv -- Series containing different divergences
    """
    # make a backup copy to store changes
    df = df.copy()

    # add result columns
    df['BULLISH_DIVERGENCE'] = 0.0
    df['BEARISH_DIVERGENCE'] = 0.0

    df['DIV_DOUB_REG_BEAR_MACD'] = 0
    df['DIV_DOUB_REG_BEAR_MACD_FROM'] = 0
    df['DIV_REG_BEAR_MACD'] = 0
    df['DIV_REG_BEAR_MACD_FROM'] = 0
    df['DIV_DOUB_REG_BULL_MACD'] = 0
    df['DIV_DOUB_REG_BULL_MACD_FROM'] = 0
    df['DIV_REG_BULL_MACD'] = 0
    df['DIV_REG_BULL_MACD_FROM'] = 0
    df['DIV_DOUB_HID_BEAR_MACD'] = 0
    df['DIV_DOUB_HID_BEAR_MACD_FROM'] = 0
    df['DIV_HID_BEAR_MACD'] = 0
    df['DIV_HID_BEAR_MACD_FROM'] = 0
    df['DIV_DOUB_HID_BULL_MACD'] = 0
    df['DIV_DOUB_HID_BULL_MACD_FROM'] = 0
    df['DIV_HID_BULL_MACD'] = 0
    df['DIV_HID_BULL_MACD_FROM'] = 0

    df['DIV_DOUB_REG_BEAR_RSI'] = 0
    df['DIV_DOUB_REG_BEAR_RSI_FROM'] = 0
    df['DIV_REG_BEAR_RSI'] = 0
    df['DIV_REG_BEAR_RSI_FROM'] = 0
    df['DIV_DOUB_REG_BULL_RSI'] = 0
    df['DIV_DOUB_REG_BULL_RSI_FROM'] = 0
    df['DIV_REG_BULL_RSI'] = 0
    df['DIV_REG_BULL_RSI_FROM'] = 0
    df['DIV_DOUB_HID_BEAR_RSI'] = 0
    df['DIV_DOUB_HID_BEAR_RSI_FROM'] = 0
    df['DIV_HID_BEAR_RSI'] = 0
    df['DIV_HID_BEAR_RSI_FROM'] = 0
    df['DIV_DOUB_HID_BULL_RSI'] = 0
    df['DIV_DOUB_HID_BULL_RSI_FROM'] = 0
    df['DIV_HID_BULL_RSI'] = 0
    df['DIV_HID_BULL_RSI_FROM'] = 0
     
    # executes divergence localization process:
    # 1. Set a default trend: requires 3 max and 3 min points
    # 1a.If max increasing or min increasing -> Bullish trend
    # 1b.If max decreasing or min decreasing -> Bearish trend
    # 1c.Else discard.

    def search(row, df, nan_value, logger):
      log = 'row [{}]: '.format(row.name)

      # check p1-p6 are valid
      if row.P6 == nan_value: 
        log += 'error-zzpoints-count'
        logger.debug(log)
        return

      # check if curr sample is max, min or the same as previous
      curr_is = 'unknown'
      # check curr sample is max 
      if row.P1 > row.P2:
        log += 'last is MAX '
        curr_is = 'max'

      #check if is min
      elif row.P1 < row.P2:
        log += 'last is MIN '
        curr_is = 'min'   

      # last 2 samples are equal, then finish
      else:
        log += 'error-no-minmax '
        logger.debug(log)
        return
      
      # at this point, exists a condition to evaluate.
      # Get idx of last 6 points (3 zigzags)
      p0_idx = row.name
      p1_idx = row.P1_idx
      p2_idx = row.P2_idx
      p3_idx = row.P3_idx
      p4_idx = row.P4_idx
      p5_idx = row.P5_idx
      p6_idx = row.P6_idx
      log += 'p0={}, p1={}, p2={}, p3={}, p4={}, p5={}, p6={} '.format(p0_idx, p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx)

      # set divergence type case
      class DivType():
        def __init__(self):
          self.enabled = False
          self.ifrom = 0
          self.to = 0          
      reg_bull_div = DivType()
      reg_bear_div = DivType()
      hid_bull_div = DivType()
      hid_bear_div = DivType()

      # Price check---
      # check regular-bearish-div
      if row.P1 > row.P3 and row.P2 > row.P4 and row.P1 > row.P2:
        reg_bear_div.enabled=True
        reg_bear_div.ifrom = row.P3_idx
        reg_bear_div.to   = row.P1_idx
      # other regular-bearish-div condition
      if row.P2 > row.P4 and row.P3 > row.P5 and row.P1 < row.P2:
        reg_bear_div.enabled=True
        reg_bear_div.ifrom = row.P4_idx
        reg_bear_div.to   = row.P2_idx
      # check hidden-bullish-div
      if row.P1 > row.P3 and row.P2 > row.P4 and row.P1 < row.P2:
        hid_bull_div.enabled=True
        reg_bear_div.ifrom = row.P3_idx
        reg_bear_div.to   = row.P1_idx
      # other hidden-bullish-div
      if row.P2 > row.P4 and row.P3 > row.P5 and row.P1 > row.P2:
        hid_bull_div.enabled=True
        reg_bear_div.ifrom = row.P4_idx
        reg_bear_div.to   = row.P2_idx
      # check regular-bullish-div
      if row.P1 < row.P3 and row.P2 < row.P4 and row.P1 < row.P2:
        reg_bull_div.enabled=True
        reg_bear_div.ifrom = row.P3_idx
        reg_bear_div.to   = row.P1_idx
      # other regular-bullish-div condition
      if row.P2 < row.P4 and row.P3 < row.P5 and row.P1 > row.P2:
        reg_bull_div.enabled=True
        reg_bear_div.ifrom = row.P4_idx
        reg_bear_div.to   = row.P2_idx
      # check hidden-bearish-div
      if row.P1 < row.P3 and row.P2 < row.P4 and row.P1 > row.P2:
        hid_bear_div.enabled=True
        reg_bear_div.ifrom = row.P3_idx
        reg_bear_div.to   = row.P1_idx
      # other hidden-bearish-div
      if row.P2 < row.P4 and row.P3 < row.P5 and row.P1 < row.P2:
        hid_bear_div.enabled=True
        reg_bear_div.ifrom = row.P4_idx
        reg_bear_div.to   = row.P2_idx

      # MACD check---
      # checking regular-bearish-div
      if reg_bear_div.enabled==True and df.MACD_main.iloc[reg_bear_div.ifrom] > df.MACD_main.iloc[reg_bear_div.to]:
        # check double divergence
        if df.ZIGZAG.iloc[reg_bear_div.ifrom+2] < df.ZIGZAG.iloc[reg_bear_div.ifrom] and df.MACD_main.iloc[reg_bear_div.ifrom+2] > df.MACD_main.iloc[reg_bear_div.ifrom]:
          log += 'doub-reg-bear-div on macd ifrom {} to {}'.format(reg_bear_div.ifrom+2, reg_bear_div.to)
          df.at[reg_bear_div.to, 'DIV_DOUB_REG_BEAR_MACD'] = 1
          df.at[reg_bear_div.to, 'DIV_DOUB_REG_BEAR_MACD_FROM'] = reg_bear_div.ifrom+2
        # else simple divergence
        else:
          log += 'reg-bear-div on macd ifrom {} to {}'.format(reg_bear_div.ifrom, reg_bear_div.to)
          df.at[reg_bear_div.to, 'DIV_REG_BEAR_MACD'] = 1
          df.at[reg_bear_div.to, 'DIV_REG_BEAR_MACD_FROM'] = reg_bear_div.ifrom
      # checking hidden-bullish-div
      if hid_bull_div.enabled==True and df.MACD_main.iloc[hid_bull_div.ifrom] > df.MACD_main.iloc[hid_bull_div.to]:
        # check double divergence
        if df.ZIGZAG.iloc[hid_bull_div.ifrom+2] < df.ZIGZAG.iloc[hid_bull_div.ifrom] and df.MACD_main.iloc[hid_bull_div.ifrom+2] > df.MACD_main.iloc[hid_bull_div.ifrom]:
          log += 'doub-hid-bull-div on macd ifrom {} to {}'.format(hid_bull_div.ifrom+2, hid_bull_div.to)
          df.at[hid_bull_div.to, 'DIV_DOUB_HID_BULL_MACD'] = 1
          df.at[hid_bull_div.to, 'DIV_DOUB_HID_BULL_MACD_FROM'] = hid_bull_div.ifrom+2
        # else simple divergence
        else:
          log += 'hid-bull-div on macd ifrom {} to {}'.format(hid_bull_div.ifrom, hid_bull_div.to)
          df.at[hid_bull_div.to, 'DIV_HID_BULL_MACD'] = 1
          df.at[hid_bull_div.to, 'DIV_HID_BULL_MACD_FROM'] = hid_bull_div.ifrom
      # checking regular-bullish-div
      if reg_bull_div.enabled==True and df.MACD_main.iloc[reg_bull_div.ifrom] < df.MACD_main.iloc[reg_bull_div.to]:
        # check double divergence
        if df.ZIGZAG.iloc[reg_bull_div.ifrom+2] > df.ZIGZAG.iloc[reg_bull_div.ifrom] and df.MACD_main.iloc[reg_bull_div.ifrom+2] < df.MACD_main.iloc[reg_bull_div.ifrom]:
          log += 'doub-reg-bull-div on macd ifrom {} to {}'.format(reg_bull_div.ifrom+2, reg_bull_div.to)
          df.at[reg_bull_div.to, 'DIV_DOUB_REG_BULL_MACD'] = 1
          df.at[reg_bull_div.to, 'DIV_DOUB_REG_BULL_MACD_FROM'] = reg_bull_div.ifrom+2
        # else simple divergence
        else:
          log += 'reg-bull-div on macd ifrom {} to {}'.format(reg_bull_div.ifrom, reg_bull_div.to)
          df.at[reg_bull_div.to, 'DIV_REG_BULL_MACD'] = 1
          df.at[reg_bull_div.to, 'DIV_REG_BULL_MACD_FROM'] = reg_bull_div.ifrom
      # checking hidden-bearish-div
      if hid_bear_div.enabled==True and df.MACD_main.iloc[hid_bear_div.ifrom] < df.MACD_main.iloc[hid_bear_div.to]:
        # check double divergence
        if df.ZIGZAG.iloc[hid_bear_div.ifrom+2] > df.ZIGZAG.iloc[hid_bear_div.ifrom] and df.MACD_main.iloc[hid_bear_div.ifrom+2] < df.MACD_main.iloc[hid_bear_div.ifrom]:
          log += 'doub-hid-bear-div on macd ifrom {} to {}'.format(hid_bear_div.ifrom+2, hid_bear_div.to)
          df.at[hid_bear_div.to, 'DIV_DOUB_HID_BEAR_MACD'] = 1
          df.at[hid_bear_div.to, 'DIV_DOUB_HID_BEAR_MACD_FROM'] = hid_bear_div.ifrom+2
        # else simple divergence
        else:
          log += 'hid-bear-div on macd ifrom {} to {}'.format(hid_bear_div.ifrom, hid_bear_div.to)
          df.at[hid_bear_div.to, 'DIV_HID_BEAR_MACD'] = 1
          df.at[hid_bear_div.to, 'DIV_HID_BEAR_MACD_FROM'] = hid_bear_div.ifrom

      # RSI check---
      # checking regular-bearish-div
      if reg_bear_div.enabled==True and df.RSI.iloc[reg_bear_div.ifrom] > df.RSI.iloc[reg_bear_div.to]:
        # check double divergence
        if df.ZIGZAG.iloc[reg_bear_div.ifrom+2] < df.ZIGZAG.iloc[reg_bear_div.ifrom] and df.RSI.iloc[reg_bear_div.ifrom+2] > df.RSI.iloc[reg_bear_div.ifrom]:
          log += 'doub-reg-bear-div on rsi ifrom {} to {}'.format(reg_bear_div.ifrom+2, reg_bear_div.to)
          df.at[reg_bear_div.to, 'DIV_DOUB_REG_BEAR_RSI'] = 1
          df.at[reg_bear_div.to, 'DIV_DOUB_REG_BEAR_RSI_FROM'] = reg_bear_div.ifrom+2
        # else simple divergence
        else:
          log += 'reg-bear-div on rsi ifrom {} to {}'.format(reg_bear_div.ifrom, reg_bear_div.to)
          df.at[reg_bear_div.to, 'DIV_REG_BEAR_RSI'] = 1
          df.at[reg_bear_div.to, 'DIV_REG_BEAR_RSI_FROM'] = reg_bear_div.ifrom
      # checking hidden-bullish-div
      if hid_bull_div.enabled==True and df.RSI.iloc[hid_bull_div.ifrom] > df.RSI.iloc[hid_bull_div.to]:
        # check double divergence
        if df.ZIGZAG.iloc[hid_bull_div.ifrom+2] < df.ZIGZAG.iloc[hid_bull_div.ifrom] and df.RSI.iloc[hid_bull_div.ifrom+2] > df.RSI.iloc[hid_bull_div.ifrom]:
          log += 'doub-hid-bull-div on rsi ifrom {} to {}'.format(hid_bull_div.ifrom+2, hid_bull_div.to)
          df.at[hid_bull_div.to, 'DIV_DOUB_HID_BULL_RSI'] = 1
          df.at[hid_bull_div.to, 'DIV_DOUB_HID_BULL_RSI_FROM'] = hid_bull_div.ifrom+2
        # else simple divergence
        else:
          log += 'hid-bull-div on rsi ifrom {} to {}'.format(hid_bull_div.ifrom, hid_bull_div.to)
          df.at[hid_bull_div.to, 'DIV_HID_BULL_RSI'] = 1
          df.at[hid_bull_div.to, 'DIV_HID_BULL_RSI_FROM'] = hid_bull_div.ifrom
      # checking regular-bullish-div
      if reg_bull_div.enabled==True and df.RSI.iloc[reg_bull_div.ifrom] < df.RSI.iloc[reg_bull_div.to]:
        # check double divergence
        if df.ZIGZAG.iloc[reg_bull_div.ifrom+2] > df.ZIGZAG.iloc[reg_bull_div.ifrom] and df.RSI.iloc[reg_bull_div.ifrom+2] < df.RSI.iloc[reg_bull_div.ifrom]:
          log += 'doub-reg-bull-div on rsi ifrom {} to {}'.format(reg_bull_div.ifrom+2, reg_bull_div.to)
          df.at[reg_bull_div.to, 'DIV_DOUB_REG_BULL_RSI'] = 1
          df.at[reg_bull_div.to, 'DIV_DOUB_REG_BULL_RSI_FROM'] = reg_bull_div.ifrom+2
        # else simple divergence
        else:
          log += 'reg-bull-div on rsi ifrom {} to {}'.format(reg_bull_div.ifrom, reg_bull_div.to)
          df.at[reg_bull_div.to, 'DIV_REG_BULL_RSI'] = 1
          df.at[reg_bull_div.to, 'DIV_REG_BULL_RSI_FROM'] = reg_bull_div.ifrom
      # checking hidden-bearish-div
      if hid_bear_div.enabled==True and df.RSI.iloc[hid_bear_div.ifrom] < df.RSI.iloc[hid_bear_div.to]:
        # check double divergence
        if df.ZIGZAG.iloc[hid_bear_div.ifrom+2] > df.ZIGZAG.iloc[hid_bear_div.ifrom] and df.RSI.iloc[hid_bear_div.ifrom+2] < df.RSI.iloc[hid_bear_div.ifrom]:
          log += 'doub-hid-bear-div on rsi ifrom {} to {}'.format(hid_bear_div.ifrom+2, hid_bear_div.to)
          df.at[hid_bear_div.to, 'DIV_DOUB_HID_BEAR_RSI'] = 1
          df.at[hid_bear_div.to, 'DIV_DOUB_HID_BEAR_RSI_FROM'] = hid_bear_div.ifrom+2
        # else simple divergence
        else:
          log += 'hid-bear-div on rsi ifrom {} to {}'.format(hid_bear_div.ifrom, hid_bear_div.to)
          df.at[hid_bear_div.to, 'DIV_HID_BEAR_RSI'] = 1
          df.at[hid_bear_div.to, 'DIV_HID_BEAR_RSI_FROM'] = hid_bear_div.ifrom
      logger.debug(log)
      return

      # is an undefined trend, then discard calculation
      log += 'error-no-trend'      
      logger.debug(log)
      #---end-of-search-function

    # execute search
    df.apply(lambda x: search(x, df, _nan_value, self.__logger), axis=1)

    # apply divergence strength
    def bullish_strength(x, df):
      return max((0.5*x.DIV_DOUB_REG_BULL_MACD) + (0.5*x.DIV_DOUB_REG_BULL_RSI) + (0.4*x.DIV_REG_BULL_MACD) + (0.4*x.DIV_REG_BULL_RSI),
                 (0.5*x.DIV_DOUB_HID_BULL_MACD) + (0.5*x.DIV_DOUB_HID_BULL_RSI) + (0.4*x.DIV_HID_BULL_MACD) + (0.4*x.DIV_HID_BULL_RSI))
    def bearish_strength(x, df):
      return max((0.5*x.DIV_DOUB_REG_BEAR_MACD) + (0.5*x.DIV_DOUB_REG_BEAR_RSI) + (0.4*x.DIV_REG_BEAR_MACD) + (0.4*x.DIV_REG_BEAR_RSI),
                 (0.5*x.DIV_DOUB_HID_BEAR_MACD) + (0.5*x.DIV_DOUB_HID_BEAR_RSI) + (0.4*x.DIV_HID_BEAR_MACD) + (0.4*x.DIV_HID_BEAR_RSI))

    df['BULLISH_DIVERGENCE'] = df.apply(lambda x: bullish_strength(x, df), axis=1)
    df['BEARISH_DIVERGENCE'] = df.apply(lambda x: bearish_strength(x, df), axis=1)
    return df


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotZigzag(self, color='black'):
    """ Plot Zigzag withing OHLC candlesticks
      Arguments:
        color -- Zigzag color (default black)
      Returns:
        [ohlc,zigzag] -- Array of traces to plot with Plot.ly
    """
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    _dfz = self.__df[self.__df.ZIGZAG > 0].copy()
    trace_zigzag = go.Scatter(x=_dfz.reset_index()['index'], y=_dfz.ZIGZAG, name='zigzag', line=scatter.Line(color=color, width=1))
    return [trace_ohlc, trace_zigzag]


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotBollinger(self, color=['black','blue','red']):
    """ Plot Bollinger indicators
      Arguments:
        color -- color (default black)
      Returns:
        [ohlc, bb_up, bb_mid, bb_lo, bb_width, bb_b] -- Array of traces to plot with Plot.ly
    """
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    trace_bollinger_up = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_HI, name='BB_up', line=scatter.Line(color=color[0], width=1))
    trace_bollinger_mid = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_MA, name='BB_ma', line=scatter.Line(color=color[0], width=1))
    trace_bollinger_down = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_LO, name='BB_lo', line=scatter.Line(color=color[0], width=1))
    trace_bollinger_width = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_WIDTH, name='BB_width', line=scatter.Line(color=color[1], width=1))
    trace_bollinger_b = go.Scatter(x=self.__df.index.values, y=self.__df.BOLLINGER_b, name='BB_%b', line=scatter.Line(color=color[2], width=1))
    return [trace_ohlc, trace_bollinger_up, trace_bollinger_mid, trace_bollinger_down, trace_bollinger_width, trace_bollinger_b]


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def plotOscillators(self, color=['blue','red','green']):
    """ Plot oscillators indicators
      Arguments:
        color -- colors
      Returns:
        [ohlc, macd_main, macd_sig, macd_hist, rsi] -- Array of traces to plot with Plot.ly
    """
    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    trace_macd_main = go.Scatter(x=self.__df.index.values, y=self.__df.MACD_main, name='MACD_main', line=scatter.Line(color=color[0], width=1))
    trace_macd_sig = go.Scatter(x=self.__df.index.values, y=self.__df.MACD_sig, name='MACD_sig', line=scatter.Line(color=color[1], width=1))
    trace_macd_hist = go.Scatter(x=self.__df.index.values, y=self.__df.MACD_hist, name='MACD_hist', line=scatter.Line(color=color[2], width=1))
    trace_rsi = go.Scatter(x=self.__df.index.values, y=self.__df.RSI, name='RSI', line=scatter.Line(color=color[0], width=1))
    return [trace_ohlc, trace_macd_main, trace_macd_sig, trace_macd_hist, trace_rsi]
            


  #-------------------------------------------------------------------
  #-------------------------------------------------------------------
  def drawIndicator(self):
    _divergences = self.__df[(self.__df.DIVERGENCE_MACD == self.__df.DIVERGENCE_RSI) & (self.__df.DIVERGENCE_MACD != 'none')][['TIME','OPEN','HIGH','LOW','CLOSE','ZIGZAG','ACTION','DIVERGENCE_MACD', 'DIVERGENCE_MACD_FROM']]

    def buildDivergenceSignal(row, df, fig):
      _from = row.DIVERGENCE_MACD_FROM
      _to = row.name  
      _trace_price = go.Scatter(x=np.array([_from,_to]), y=np.array([df.ZIGZAG[_from], df.ZIGZAG[_to]]), line=scatter.Line(color='blue', width=1))
      fig.append_trace(_trace_price, 1, 1)
      _trace_macd = go.Scatter(x=np.array([_from,_to]), y=np.array([df.MACD_main[_from], df.MACD_main[_to]]), line=scatter.Line(color='black', width=1))
      fig.append_trace(_trace_macd, 2, 1)
      _trace_rsi = go.Scatter(x=np.array([_from,_to]), y=np.array([df.RSI[_from], df.RSI[_to]]), line=scatter.Line(color='black', width=1))
      fig.append_trace(_trace_rsi, 3, 1)

    # Plot ohlc,zigzag, MACD and RSI
    # setup plotting figure with 3 rows and 1 column
    fig = plotly.tools.make_subplots(rows=3, cols=1, subplot_titles=('Price', 'Oscillators'), shared_xaxes=True, vertical_spacing=0.1)

    trace_ohlc = go.Ohlc(x=self.__df.index.values, open=self.__df.OPEN, high=self.__df.HIGH, low=self.__df.LOW, close=self.__df.CLOSE, name='Candlestick')
    fig.append_trace(trace_ohlc, 1, 1)

    _dfz = self.__df[self.__df.ZIGZAG > 0].copy()
    trace_zigzag = go.Scatter(x=_dfz.reset_index()['index'], y=_dfz.ZIGZAG, name='zigzag', line=scatter.Line(color='black', width=1))
    fig.append_trace(trace_zigzag, 1, 1)

    trace_macd = go.Scatter(x=self.__df.index.values, y=self.__df.MACD_main, name='macd', line=scatter.Line(color='blue', width=1))
    fig.append_trace(trace_macd, 2, 1)

    trace_rsi = go.Scatter(x=self.__df.index.values, y=self.__df.RSI, name='rsi', line=scatter.Line(color='red', width=1))
    fig.append_trace(trace_rsi, 3, 1)

    # add signals of divergence to both oscillators and price
    _divergences.apply(lambda x: buildDivergenceSignal(x, self.__df, fig), axis=1)

    fig['layout'].update(height=600, title='Divergences')

    # reference result
    self.__fig = fig
    return self.__fig

