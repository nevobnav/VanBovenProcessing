# SECURITY WARNING: keep the secret key used in production secret!
# testing github
import os
import json
from datetime import datetime
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

def insert_new_scan(date, time, plot, meta, con, zoomlevel = 23, flight_altitude=35):
    try:
        int(plot)
        plot_id = plot
    except:
        plot_id = get_plot_id(plot,meta,con)
    new_scan = {'date': date,
                'time': time,
                'plot_id': plot_id,
                'zoomlevel': zoomlevel,
                'flight_altitude': flight_altitude
                }
    scans = meta.tables['portal_scan']
    insert_new_scan = scans.insert().values(new_scan)
    con.execute(insert_new_scan)

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
