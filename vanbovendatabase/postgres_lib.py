# SECURITY WARNING: keep the secret key used in production secret!
# testing github
import os
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, MetaData, Integer, ForeignKey, DateTime, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
from geoalchemy2 import Geometry


def connect(user, password, db, host='localhost', port=5432):
    '''Returns a connection and a metadata object'''
    # We connect with the help of the PostgreSQL URL
    # postgresql://federer:grandestslam@localhost:5432/tennis
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)

    # The return value of create_engine() is our connection object
    con = create_engine(url, client_encoding='utf8')

    # We then bind the connection to MetaData()
    meta = MetaData(bind=con, reflect=True)

    return con, meta


def insert_new_scan(meta, con, date, time, plot, no_of_images = 0, zoomlevel = 23, flight_altitude=35, sensor='unknown', quality='MED', upload_time=None,
                    preprocess_time=None, ortho_time = None, tiles_time = None, live=False, seen_by_user=False):
    try:
        int(plot)
        plot_id = plot
    except:
        plot_id = get_plot_id(plot,meta,con)
    new_scan = {'date': date,
                'time': time,
                'plot_id': plot_id,
                'no_imgs': no_of_images,
                'zoomlevel': zoomlevel,
                'flight_altitude': flight_altitude,
                'sensor': sensor,
                'quality': quality,
                'upload': upload_time,
                'preprocess': preprocess_time,
                'ortho': ortho_time,
                'tiles': tiles_time,
                'live': live,
                'seen_by_user': seen_by_user
                }
    scans = meta.tables['portal_scan']
    insert_new_scan = scans.insert().values(new_scan)
    con.execute(insert_new_scan)

    query= select([scans.c.id])
    query = query.where(scans.c.date == date)
    query = query.where(scans.c.time == time)
    query = query.where(scans.c.plot_id == plot_id)
    res = con.execute(query)
    new_scan_id = res.fetchall()[0][0]

    return new_scan_id

def update_scan(con,meta,update_dict,plot_id=None,date=None,time=None,scan_id=None):
    #Usage example: update_scan(con,meta,'Wever oost','2019-03-21','02:00:00',{'zoomlevel':23})
    scans = meta.tables['portal_scan']
    if scan_id:
        update_me = scans.update().\
            where(scans.c.id == scan_id).\
            values(update_dict)
    else:
        if type(plot_id) is str:
            plot_id = get_plot_id(plot_id,meta,con)

        update_me = scans.update().\
            where(scans.c.plot_id == plot_id).\
            where(scans.c.date == date).\
            where(scans.c.time == time).\
            values(update_dict)
    response = con.execute(update_me)
    return

def get_scan(con, meta, date=None, time=None, plot=None, scan_id=None):
    scans = meta.tables['portal_scan']

    query = select(['*'])
    if scan_id:
        query = query.where(scans.c.id == scan_id)
    else:
        if type(plot) is str:
            plot = get_plot_id(plot,meta,con)
        #check for time within 5 minute timeframe
        time_object = datetime.strptime(time, '%H%M')
        delta = timedelta(minutes = 5)

        time_max = (time_object+delta).time()
        time_min = (time_object-delta).time()

        query = query.where(scans.c.date == date)
        query = query.where(scans.c.time >= time_min)
        query = query.where(scans.c.time <= time_max)
        query = query.where(scans.c.plot_id == plot)

    res = con.execute(query)

    result_dicts = []
    for row in res.fetchall():
        res_line = {'scan_id': row[0], 'date':row[1], 'plot_id':row[2], 'time':row[3], 'zoomlevel':row[4],
                    'flight_altitude':row[5], 'live':row[6], 'no_imgs':row[7], 'ortho':row[8],
                   'precprocess':row[9], 'quality':row[10], 'sensor':row[11], 'tiles':row[12], 'upload':row[13]}
        result_dicts.append(res_line)

    return result_dicts

def get_customer_pk(customer_name,meta,con):
    customer_pk = None
    customers = meta.tables['portal_customer']
    query = select([customers.c.id])
    query = query.where(customers.c.customer_name == customer_name)
    res = con.execute(query)
    for result in res:
        customer_pk = result[0]
    if customer_pk:
        return customer_pk
    else:
        return None


def get_customer_plots(customer_name, meta, con):
    customer_pk = get_customer_pk(customer_name,meta,con)
    plot_id = None
    #First grab all parent_plot ids asociated with this customer_pk
    parent_plots = meta.tables['portal_parent_plot']
    query= select([parent_plots.c.id,parent_plots.c.name])
    query = query.where(parent_plots.c.customer_id == customer_pk)
    res = con.execute(query).fetchall()
    parent_plot_ids = [x[0] for x in res]
    parent_plot_names = [x[1] for x in res]

    #Now build list of the plots themselves with these parent_plots:
    plots = meta.tables['portal_plot']
    plot_ids = []
    for parent_plot_id in parent_plot_ids:
        query= select([plots.c.id])
        query = query.where(plots.c.parent_plot_id == parent_plot_id)
        res = con.execute(query).fetchall()
        plot_ids.append([x[0] for x in res][0])
    return plot_ids, parent_plot_names


def get_plot_customer(plotname, meta, con):
    parent_plots = meta.tables['portal_parent_plot']
    query= select([parent_plots.c.customer_id])
    query = query.where(parent_plots.c.name == plotname)
    try:
        customer_id = con.execute(query).first()[0]
        customer_name = get_customer_name(customer_id,meta,con)
        return customer_name
    except:
        return []


def get_customer_name(pk,meta,con):
    customers = meta.tables['portal_customer']
    query= select([customers.c.customer_name])
    query = query.where(customers.c.id == pk)
    try:
        res = con.execute(query).first()[0]
    except:
        res = []
    return res


def get_plot_shape(plot, meta,con):
    try:
        int(plot)
        plot_id = plot
    except:
        plot_id = get_plot_id(plot,meta,con)

    plots = meta.tables['portal_plot']
    query= select([plots.c.shape])
    query = query.where(plots.c.id == plot_id)
    res = con.execute(query)
    for result in res:
        output = result[0]
    return output

def get_plot_id(plot_name, meta,con):
    #get parent_plot_id
    parent_plots = meta.tables['portal_parent_plot']
    query= select([parent_plots.c.id])
    query = query.where(parent_plots.c.name == plot_name)
    res = con.execute(query)
    for result in res:
        parent_plot_id = result[0]

    #determine plot_id from paren_plot_id (reverse child, sucks):
    plots = meta.tables['portal_plot']
    query= select([plots.c.id])
    query = query.where(plots.c.parent_plot_id == parent_plot_id)
    res = con.execute(query).fetchall()
    plot_id = res[0][0]
    return plot_id



def get_scans_by_date(date, meta,con):
    scans = meta.tables['portal_scan']
    query= select()
    query = query.where(scans.c.date == date)
    res = con.execute(query)
    outputs = []
    for result in res:
        new_val = result[0]
        outputs.append(new_val)
    return outputs
