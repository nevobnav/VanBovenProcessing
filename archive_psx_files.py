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
import time
import logging
import multiprocessing

#root processing jobs path
root_processing_path = r"O:/SfM_Jobs/"

if multiprocessing.cpu_count() <32:
    metashape_path = r'E:\Metashape'
elif multiprocessing.cpu_count() => 32:
    metashape_path = r'D:\200 Metashape'
    
output_path = r'O:\900 Metashape archive'
days_to_store = 14

#initiate log file
timestr = time.strftime("%Y%m%d-%H%M%S")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename = os.path.join(root_processing_path, "Log_files/" + str(timestr) + "_Metashape_archiving_log_file.log"),level=logging.DEBUG)


def find_old_psx_files(metashape_path, days_to_store, output_path):
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
                            print('archiving ' + file + ' ...')
                            psx_file = os.path.join(i, file)
                            #Metashape.app.console.clear() before version 1.6
                            Metashape.app.console_pane.clear()
                            doc = Metashape.app.document
                            doc.open(psx_file)
                            pszfile = psx_file[:-4]+'.psz'

                            out_path = os.path.dirname(pszfile)
                            if os.path.exists(out_path) == False:
                                os.makedirs(out_path)

                            pszfile = pszfile.replace(metashape_path, output_path)

                            doc.save(pszfile )
                            os.remove(psx_file)
                            file_dir = psx_file[:-4]+'.files'
                            shutil.rmtree(file_dir)
                        else:
                            continue
                    except:
                        logging.exception("Metashape processing encountered the following problem:")
                        logging.info("\n")
                        print('error encountered while archiving')

find_old_psx_files(metashape_path, days_to_store, output_path)
