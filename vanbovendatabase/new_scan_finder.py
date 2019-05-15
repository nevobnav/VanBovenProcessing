import os
from datetime import datetime
from postgres_lib import *

with open('/Users/kazv/Documents/Programming/VanBoven/VanBovenDatabase/postgis_config.json') as config_file:
    config = json.load(config_file)

DB_NAME = config['NAME']
DB_USER = config['DB_USER']
DB_PASSWORD = config['DB_PASSWORD']
DB_IP = config['DB_IP']
con,meta = connect(DB_USER, DB_PASSWORD, DB_NAME, host=DB_IP)

data_path = '/Users/kazv/Documents/Programming/VanBoven/MijnVanBoven/media/data'

walk = os.walk(data_path)
subdirs = [x[0] for x in os.walk(data_path)]

now_str = datetime.now().strftime('%Y%m%d')
now_str = '20190501'

today_dirs = [x for x in subdirs if now_str in x]
if len(today_dirs)> 0:
    data_slash = today_dirs[0].find('/data') + 5
new_scans = []


for dir in today_dirs:
    if len(today_dires)>0:
        slash_positions = [pos for pos,char in enumerate(dir) if char == "/"]
        slashes_after_data = [x for x in slash_positions if x > data_slash]
        if len(slashes_after_data) == 2:
            customer_slash = slashes_after_data[0]
            plot_slash = slashes_after_data[1]
            customer = dir[data_slash+1 : customer_slash]
            plot = dir[customer_slash+1 : plot_slash]
            dict = {'customer':customer, 'plot':plot}
            new_scans.append(dict)
        else:
            break
    else:
        break

print(new_scans)

#Fetch existing scans for today:
scans = get_scans_by_date(now_str, meta,con)
print(scans)
