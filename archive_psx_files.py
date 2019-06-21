# -*- coding: utf-8 -*-
"""
Created on Tue May  7 08:57:38 2019

@author: VanBoven
"""

import os
import datetime
import pandas as pd
import shutil
import Metashape

metashape_path = r'E:\Metashape'
output_path = r'F:\Metashape'
days_to_store = 14

def find_old_psx_files(metashape_path, days_to_store):
    today = datetime.date.today()
    threshold_date = today - datetime.timedelta(days=days_to_store)
    folderList = pd.DataFrame([x[0] for x in os.walk(metashape_path)], columns = ['Path'])
    for i in folderList.Path:
        if len(i.split('\\')) == 4:
            files = os.listdir(i)
            for file in files:
                if file.endswith('.psx'):
                    try:
                        file_created_str = file[:8]
                        file_created_date = datetime.date(int(file_created_str[:4]),int(file_created_str[4:6]), int(file_created_str[6:]))
                        if file_created_date < threshold_date:
                            psx_file = os.path.join(i, file)
                            out_path = 'F'+i[1:]
                            if os.path.exists(out_path) == False:
                                os.makedirs(out_path)                                                           
                            Metashape.app.console.clear()
                            doc = Metashape.app.document
                            doc.open(psx_file)
                            pszfile = psx_file[1:-4]+'.psz'
                            doc.save( 'F'+pszfile )
                            os.remove(psx_file)
                            file_dir = psx_file[:-4]+'.files'
                            shutil.rmtree(file_dir)
                        else:
                            continue
                    except:
                        print('error encountered while archiving')

find_old_psx_files(metashape_path, days_to_store)
