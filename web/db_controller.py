from credentials import credentials_data
import select
import psycopg2
import psycopg2.extensions
from threading import Lock 
psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)
from gevent import monkey
monkey.patch_time()

DB_CONN_STRING = "host='%s' dbname='%s' user='%s' password='%s'" \
    % (credentials_data['db_ip'],credentials_data['db_name'],credentials_data['db_user'],credentials_data['db_password'])

socketio = None
thread = None
thread_lock = Lock()


# Wait for other connections to close
def db_wait(conn):
    while True:
        state = conn.poll()
        #print("STATE %s" % state)

        if state == psycopg2.extensions.POLL_OK:
            #print("\tPOLL_OK")
            break
        elif state == psycopg2.extensions.POLL_WRITE:
            #print("\tPOLL_WRITE")
            select.select([], [conn.fileno()], [])
        elif state == psycopg2.extensions.POLL_READ:  
            #print("\tPOLL_READ")
            select.select([conn.fileno()], [], [])
        else:
            raise psycopg2.OperationalError("poll() returned %s" % state)

        socketio.sleep(0.01)

    # Print any notices
    notice_data = [{'data': notice} for notice in conn.notices]
    print("#### Notice::%s"%notice_data)
    if (notice_data != [] or len(notice_data) > 0):
        print("     db_wait: sending notice data 1:%s, 2:%s" % (notice_data == [], len(notice_data)))
        socketio.emit('message_event', notice_data, namespace='/mldbns')


# Return _async notices then close connection
def db_wait_close(conn, cur, close):
    global thread
    global thread_lock
    rows = None

    while True:
        state = conn.poll()
        #print("STATE %s" % state)

        # Print any notices
        while conn.notices:
            notice = conn.notices.pop(0)
            socketio.emit('message_event', {'data': notice}, namespace='/mldbns')
            print("#### Notice::%s"%notice)

        if state == psycopg2.extensions.POLL_OK:
            #print("\tPOLL_OK")
            try:
                rows = cur.fetchall()
            except Exception as e:
                pass # query could not return rows but not an error
            break
        elif state == psycopg2.extensions.POLL_WRITE:
            #print("\tPOLL_WRITE")
            select.select([], [conn.fileno()], [])
        elif state == psycopg2.extensions.POLL_READ:  
            #print("\tPOLL_READ")
            select.select([conn.fileno()], [], [])
        else:
            raise psycopg2.OperationalError("poll() returned %s" % state)

        socketio.sleep(0.001)

    if close:
        # Close DB objects
        print('Closing async DB connection')
        if cur:
            cur.close()
        if conn:
            conn.close()

        # Reset thread
        thread = None
        thread_lock = Lock()
    
    return rows


# Execute query
def db_execute(_socketio, query, query_values, _async):
    global socketio
    global thread
    global thread_lock
    socketio = _socketio
    conn = None
    cur = None
    rows = None

    print("Attempting to exec (%s) query [%s] with params %s" % ('async' if _async else 'sync',query,query_values))

    try:
        conn = psycopg2.connect(DB_CONN_STRING, async_=_async)
        if (_async):
            db_wait(conn)
        else:
            #conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            conn.set_isolation_level(0)
        cur = conn.cursor() 

        cur.execute(query, query_values)
        if (_async):
            rows = db_wait_close(conn, cur, True)
        else:
            # Get any returned data
            try:
                rows = cur.fetchall()
            except Exception as e:
                pass # query could not return rows but not an error

            # Print any notices
            notice_data = [{'data': notice} for notice in conn.notices]
            print("#### Notice::%s"%notice_data)
            if (notice_data != [] or len(notice_data) > 0):
                print("     db_execute: sending notice data 1:%s, 2:%s" % (notice_data == [], len(notice_data)))
                socketio.emit('message_event', notice_data, namespace='/mldbns')
            
            print("Closing sync DB connection")

            if cur:
                cur.close()
            if conn:
                conn.close()

            # Reset thread
            thread = None
            thread_lock = Lock()

        socketio.emit('message_event', {'data' : 'DB Activity finished successfully'}, namespace='/mldbns')

    except psycopg2.DatabaseError as e:
        print('! psycopg2 Error: %s' % e)
        print('Error-Closing DB connection')
        socketio.emit('message_event', {'data' : 'DB ERROR: %s' % e}, namespace='/mldbns')

        if cur:
            cur.close()
        if conn:
            conn.close()

        # Reset thread
        thread = None
        thread_lock = Lock()
    except Exception as e:
        print('! DB Exception: %s' % e)
        socketio.emit('message_event', {'data' : 'EXCEPTION: %s' % e}, namespace='/mldbns')

        if cur:
            cur.close()
        if conn:
            conn.close()

        # Reset thread
        thread = None
        thread_lock = Lock()
    """
    finally:
        print('Closing DB connection')
        if cur:
            cur.close()
        if conn:
            conn.close()
    """

    return rows