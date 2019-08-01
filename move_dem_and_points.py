# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:00:54 2019

@author: VanBoven
"""

import os
import shutil

ortho_archive_destination = r'D:\VanBovenDrive\VanBoven MT\Archive' #Folder where orthos are archived (gdrive)
working_dir = r'C:\Users\VanBoven\Documents\100 Ortho Inbox\ready' #Refer to path where DEM and .points file are being used for gereferencing

def move_DEM_and_pointsfile(ortho_archive_destination, working_dir):
    files = os.listdir(working_dir)
    
