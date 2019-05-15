# SECURITY WARNING: keep the secret key used in production secret!

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

def insert_new_scan(date, time, plot, meta, con):
    try:
        int(plot)
        plot_id = plot
    except:
        plot_id = get_plot_id(plot,meta,con)
    new_scan = {'date': date,
                'time':time,
                'plot_id': plot_id
                }
    scans = meta.tables['portal_scan']
    insert_new_scan = scans.insert().values(new_scan)
    con.execute(insert_new_scan)

def get_customer_pk(customer_name,meta,con):
    customers = meta.tables['portal_customer']
    query = select([customers.c.id])
    query = query.where(customers.c.customer_name == customer_name)
    res = con.execute(query)
    for result in res:
        customer_pk = result[0]
    return customer_pk

def get_customer_name(customer_pk,meta,con):
    customers = meta.tables['portal_customer']
    query = select([customers.c.customer_name])
    query = query.where(customers.c.id == customer_pk)
    res = con.execute(query)
    for result in res:
        customer_pk = result[0]
    return customer_pk

def get_customer_plots(customer_name, meta, con):
    customer_pk = get_customer_pk(customer_name,meta,con)
    plots = meta.tables['portal_plot']
    query= select([plots.c.id])
    query = query.where(plots.c.customer_id == customer_pk)
    res = con.execute(query)
    plot_ids = []
    for result in res:
        new_val = result[0]
        plot_ids.append(new_val)
    return plot_ids

def get_customer_plot_names(customer_name, meta, con):
    customer_pk = get_customer_pk(customer_name,meta,con)
    plots = meta.tables['portal_plot']
    query= select([plots.c.id])
    query = query.where(plots.c.customer_id == customer_pk)
    res = con.execute(query)
    plot_names = []
    for result in res:
        new_val = result[0]
        plot_names.append(new_val)
    return plot_names

def get_plot_customer(plotname, meta, con):
    plots = meta.tables['portal_plot']
    query= select([plots.c.customer_name])
    query = query.where(plots.c.name == plotname)
    res = con.execute(query)
    for result in res:
        customer_name = result[0]
    return customer_name

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
    plots = meta.tables['portal_plot']
    query= select([plots.c.id])
    query = query.where(plots.c.name == plot_name)
    res = con.execute(query)
    for result in res:
        output = result[0]
    return output

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
