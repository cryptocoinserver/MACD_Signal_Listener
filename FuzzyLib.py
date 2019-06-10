#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  FuzzyLib is a python class that provides basic Fuzzy Logic operations: 
  - Fuzzification
  - Defuzzification
  
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

class Fuzzifier():
  def fuzzify(crisp_value, fuzzy_sets): 
    """ Fuzzifies a crisp values according with some group points

      Keyword arguments:
        crisp_value -- Value to fuzzify
        fuzzy_set_points -- Sets points (2 points for extremums sets and 3 points for rest)
      Returns:
        u[] -- memberships to sets
    """
    def _line(x0,x1,y0,y1,x):
      return (((y1-y0)/(x1-x0))*(x-x0))+y0
    _fuzzy_values = []    
    for s in fuzzy_sets:      
      u = 0.0
      # first set
      if s['type']=='left-edge':
        if crisp_value <= s['p0']:
          u = 1.0
        elif crisp_value >= s['p1']:
          u = 0.0
        else:
          u = _line(s['p0'],s['p1'],1.0,0.0,crisp_value)

      # last set
      elif s['type']=='right-edge':
        if crisp_value >= s['p1']:
          u = 1.0
        elif crisp_value <= s['p0']:
          u = 0.0
        else:
          u = _line(s['p0'],s['p1'],0.0,1.0,crisp_value)
  
      # intermediate sets
      else:
        if s['type'] == 'internal-4pt':
          if crisp_value >= s['p3'] or crisp_value <= s['p0']:
            u = 0.0
          elif crisp_value <= s['p1']:
            u = _line(s['p0'],s['p1'],0.0,1.0,crisp_value)
          elif crisp_value >= s['p2']:
            u = _line(s['p2'],s['p3'],1.0,0.0,crisp_value)
          else:
            u = 1.0
        elif s['type'] == 'internal-3pt':
          if crisp_value >= s['p2'] or crisp_value <= s['p0']:
            u = 0.0
          elif crisp_value <= s['p1']:
            u = _line(s['p0'],s['p1'],0.0,1.0,crisp_value)
          else:
            u = _line(s['p1'],s['p2'],1.0,0.0,crisp_value)

      _fuzzy_values.append(u)
    return np.asarray(_fuzzy_values)


####################################################################################
####################################################################################
####################################################################################

class FuzzyVar():
  def __init__(self, name, fuzzy_set_points, level=logging.WARN):
    self.__logger = logging.getLogger(__name__)
    self.__logger.setLevel(level)
    self.__logger.info('Created!')
    self.__name = name
    self.__fuzzy_set_points = fuzzy_set_points
    self.__num_sets = int((len(fuzzy_set_points)+2)/3)
    self.__crisp_value = 0.0
    self.__fuzzy_values = []
    for i in range(self.__num_sets):
      self.__fuzzy_values.append(0.0)

  
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
  def fuzzify(self, crisp_value): 
    """ Fuzzifies a crisp value

      Keyword arguments:
        crisp_value -- Value to fuzzify
      Returns:
        u[] -- memberships to sets
    """
    self.__crisp_value = crisp_value
    self.__fuzzy_values = Fuzzifier.fuzzify(self.__crisp_value, self.__fuzzy_set_points)
    return self.__fuzzy_values


  