# coding: utf-8

import pika


def get_queue_connection_params(params):
    # the tcp_opts below enable TCP keepalive; tune it if needed
    tcp_opts = {
        'TCP_KEEPIDLE': 60,
        'TCP_KEEPINTVL': 2,
        'TCP_KEEPCNT': 2,
    }
    usern = params['username']
    passw = params['password']
    if usern and passw:
        creds = pika.PlainCredentials(username=usern, password=passw)
        conn_params = pika.ConnectionParameters(host=params['host'],
                                                port=params['port'],
                                                tcp_options=tcp_opts,
                                                heartbeat=params['heartbeat'],
                                                credentials=creds)
    else:
        conn_params = pika.ConnectionParameters(host=params['host'],
                                                port=params['port'],
                                                tcp_options=tcp_opts,
                                                heartbeat=params['heartbeat'])

    return conn_params


def get_pika_conn_blocking(params_rabbit):
    conn_params = get_queue_connection_params(params_rabbit)
    connection = pika.BlockingConnection(conn_params)
    return connection
