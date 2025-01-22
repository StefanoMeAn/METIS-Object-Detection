# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:48:11 2021

@author: chioettop
"""

import pandas as pd
import numpy as np

class StarCatalog():
    
    def __init__(self, catalog='Simbad'):
        if catalog == 'BSC5':
            # reads the BSC5 Bright Stars Catalogue
            colspecs = [(0,4), (4,14), (25,31), (41,42), (43,44), (51,60),
                        (75, 77), (77, 79), (79, 83), (83, 84), (84, 86), (86, 88), (88, 90), 
                        (90,96), (96,102), (102,107), (109,114), (115,120), (127,147),
                        (148, 154), (154, 160)]
    
            labels = ['HR', 'Name', 'HD', 'IRflag', 'Multiple', 'VarID',
                      'RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs',
                      'GLON', 'GLAT', 'Vmag', 'B-V', 'U-B', 'SpType',
                      'pmRA', 'pmDE']
    
            self._cat = pd.read_fwf('bsc5.dat', header= None, colspecs=colspecs,
                        names=labels, index_col=0)
            
            # remove rows with Nan on the HD
            self._cat.dropna(subset=['HD'], inplace=True)
            
            # convert ra, dec to degrees
            b = self._cat
            self._cat['RAdeg'] = (b.RAh + b.RAm/60 + b.RAs/3600) * 15
            sign = b['DE-'].replace('-', -1).replace('+', 1)
            self._cat['DEdeg'] = (b.DEd + b.DEm/60 + b.DEs/3600) * sign 
            
            self._cat.HD = 'HD' + self._bsc.HD.astype('int').astype('str')
            
            self._cat.rename(columns={'RAdeg': 'ra', 'DEdeg': 'dec'})
            self._cat['MAIN_ID'] = self._cat.Name.fillna(self._cat.HD)
            
        elif catalog == 'Simbad':
            """
            read the Simbad catalog saved on 09/04/2021 with the following query
            votable vot1 {MAIN_ID, 
                          RA(d;A;ICRS;J2000;2000), DEC(d;D;ICRS;J2000;2000),
    	                  PMRA, 	PMDEC, FLUX(V), FLUX(U)}
            votable open vot1
            query sample Vmag <= 7
            votable close 
            """
            self._cat = pd.read_csv('simbad_vmag7.csv')
        
    def query(self, ra, dec, r_max, r_min=0):
        # Returns stars within a circle centered in ra, dec
        # and of radius r. Assumes ra, dec in J2000 at epoch 2000
        
        # angular distance: cosθ=sinδ1sinδ2+cosδ1cosδ2cos(α1−α2)
        
        a1 = np.deg2rad(self._cat.ra)
        a2 = np.deg2rad(ra)
        d1 = np.deg2rad(self._cat.dec)
        d2 = np.deg2rad(dec)
        
        adist = np.arccos(
            np.sin(d1)*np.sin(d2)+np.cos(d1)*np.cos(d2)*np.cos(a1-a2)
            )
        
        inside = (adist <= np.deg2rad(r_max)) & (adist >= np.deg2rad(r_min))
        
        return self._cat[inside].copy()
            
            
            