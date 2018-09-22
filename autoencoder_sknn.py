#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 12:24:49 2018

@author: abhi
"""

import sklearn   
import sknn.ae 

sknn.ae.AutoEncoder(layers, warning=None, parameters=None, random_state=None, learning_rule=u'sgd', learning_rate=0.01, learning_momentum=0.9, normalize=None, regularize=None, weight_decay=None, dropout_rate=None, batch_size=1, n_iter=None, n_stable=10, f_stable=0.001, valid_set=None, valid_size=0.0, loss_type=None, callback=None, debug=False, verbose=None, **params)

sknn.ae.Layer(activation, warning=None, type=u'autoencoder', name=None, units=None, cost=u'msre', tied_weights=True, corruption_level=0.5)
