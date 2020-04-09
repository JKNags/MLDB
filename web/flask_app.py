#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import select
#from OpenSSL import SSL

from threading import Lock 
from flask import Flask, render_template, request, send_file, send_from_directory, make_response
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO # as emit
from werkzeug.utils import secure_filename
from gevent import monkey
monkey.patch_time()

app = Flask(__name__)    # Construct an instance of Flask class
CORS(app)                # apply CORS

from db_controller import db_execute
#import flask_ml # brokeded


# Function to check if a variable can be cast to a number
def is_numeric(x):
    try: 
        float(x)
        return True
    except ValueError:
        return False


######################
# Socket IO
######################

socketio = SocketIO(app, _async_mode='gevent', logger=True, engineio_logger=True, ping_timeout=60)

@socketio.on('connect', namespace='/mldbns')
def socket_connect():
    emit('message_event', {'data':'Connected to Server'} )

    print("Client connected to mldbns server")

@socketio.on('button_event', namespace='/mldbns')
def socket_message(message):
    emit('message_event', {'data':'Beginning DB Query - %s' % message['type']} )
    print("Client issued button event type=(%s) - data: %s" % (message['type'],message['data']))

    try:
        error_message = None
        _async = False   # db connection async boolean
        query = '' # query prepared for string formatting (only %s)
        query_values = [] # array or tuple of parameter values
        data = message['data']

        if(message['type'] == 'test'):
            _async = True
            query = 'call sp_tmp(%s)'
            query_values = [data]

            db_execute(socketio, query, query_values, _async)

        elif(message['type'] == 'refresh_dataset'):
            _async = False
            query = """
                SELECT  d.id,
                        d.name
                FROM dataset d
            """
            query_values = None
            dataset_list = []
            
            rows = db_execute(socketio, query, query_values, _async)
            if rows is not None:
                for row in rows:
                    try:                                     
                        dataset_list.append({'id':row[0],
                                            'name':row[1]})
                    except Exception:
                        pass
                emit('refresh_dataset_event', {'data':dataset_list})
            else:
                print('Refresh Dataset rows is None')    

        elif(message['type'] == 'refresh_network'):
            _async = False
            query = """
                SELECT  nna.id,
                        nna.dataset_id,
                        nna.activation_functions,
                        nna.loss_function,
                        nna.name,
                        nna.num_hidden_layers,
                        nna.layer_sizes,
                        d.name AS dataset_name
                FROM neural_network_architecture nna
                JOIN dataset d ON nna.dataset_id = d.id
                ORDER BY id DESC
            """
            query_values = None
            network_list = []
            
            rows = db_execute(socketio, query, query_values, _async)
            if rows is not None:
                for row in rows:
                    try:
                        activation_functions = ['' if len(row[2]) < 1 else row[2][0],
                                                '' if len(row[2]) < 2 else row[2][1],
                                                '' if len(row[2]) < 3 else row[2][2]]
                        layer_sizes_arr = ['' if len(row[6]) < 1 else row[6][0],
                                                '' if len(row[6]) < 2 else row[6][1],
                                                '' if len(row[6]) < 3 else row[6][2]]
                        layer_sizes_str = 'IN:%d, HDN:%dx%d, OUT:%d' % (layer_sizes_arr[0],row[5],layer_sizes_arr[1],layer_sizes_arr[2])                                        
                        network_list.append({'id':row[0],
                                            'name':row[4],
                                            'activation_functions':activation_functions,
                                            'loss_function':row[3],
                                            'layer_sizes':layer_sizes_str,
                                            'dataset_name':row[7]})
                    except Exception:
                        pass
                emit('refresh_network_event', {'data':network_list})
            else:
                print('Refresh Network rows is None')    
            

        elif(message['type'] == 'reset_network'):
            _async = False
            
            # Input Validations
            while (True): # loop just to break from it on first error
                if (data['network_id'] is None or data['network_id'] == ''):
                    error_message = 'INPUT VALIDATION: Network ID is required.'; break

                if (int(data['network_id']) <= 0):
                    error_message = 'INPUT VALIDATION: Network ID must be greater than 0.'; break

                query = 'call sp_reset_network(%s,%s)'
                query_values = [data['network_id'], False]

                break #end of validations

            if (error_message is not None):
                emit('message_event', {'data' : error_message} )
            elif (query == ""):
                emit('message_event', {'data' : 'Cannot execute empty query'} )
            else:
                db_execute(socketio, query, query_values, _async)

        elif(message['type'] == 'delete_dataset'):
            _async = False

            # Input Validations
            while (True): # loop just to break from it on first error
                if (data['dataset_id'] is None or data['dataset_id'] == ''):
                    error_message = 'INPUT VALIDATION: dataset ID is required.'; break

                if (int(data['dataset_id']) <= 0):
                    error_message = 'INPUT VALIDATION: dataset ID must be greater than 0.'; break

                query = 'call sp_delete_dataset(%s)'
                query_values = [data['dataset_id']]

                break #end of validations

            if (error_message is not None):
                emit('message_event', {'data' : error_message} )
            elif (query == ""):
                emit('message_event', {'data' : 'Cannot execute empty query'} )
            else:
                db_execute(socketio, query, query_values, _async)

        elif(message['type'] == 'delete_network'):
            _async = False

            # Input Validations
            while (True): # loop just to break from it on first error
                if (data['network_id'] is None or data['network_id'] == ''):
                    error_message = 'INPUT VALIDATION: Network ID is required.'; break

                if (int(data['network_id']) <= 0):
                    error_message = 'INPUT VALIDATION: Network ID must be greater than 0.'; break

                query = 'call sp_delete_network(%s)'
                query_values = [data['network_id']]

                break #end of validations

            if (error_message is not None):
                emit('message_event', {'data' : error_message} )
            elif (query == ""):
                emit('message_event', {'data' : 'Cannot execute empty query'} )
            else:
                db_execute(socketio, query, query_values, _async)
                socket_message({'type':'refresh_network', 'data':''}) # force refresh

        elif(message['type'] == 'insert_dataset'):
            _async = False

            # Input Validations
            while (True): # loop just to break from it on first error
                if (data['dataset_name'] is None or data['dataset_name'] == ''):
                    error_message = 'INPUT VALIDATION: Dataset Name is required.'; break
                if (data['dataset_source_table_name'] is None or data['dataset_source_table_name'] == ''):
                    error_message = 'INPUT VALIDATION: Source Table Name is required.'; break
                if (data['feature_column_names'] is None or data['feature_column_names'] == ''):
                    error_message = 'INPUT VALIDATION: Feature Column Names is required.'; break
                if (data['label_column_names'] is None or data['label_column_names'] == ''):
                    error_message = 'INPUT VALIDATION: Label Column Names is required.'; break
                
                query = 'call sp_insert_dataset(%s,%s,%s,%s)'
                query_values = [
                    data['dataset_name'],
                    data['dataset_source_table_name'],
                    data['feature_column_names'],
                    data['label_column_names'],
                    data['normalize_features'],
                    data['normalize_labels']
                ]

                break #end of validations

            if (error_message is not None):
                emit('message_event', {'data' : error_message} )
            elif (query == ""):
                emit('message_event', {'data' : 'Cannot execute empty query'} )
            else:
                db_execute(socketio, query, query_values, _async)
                socket_message({'type':'refresh_dataset', 'data':''}) # force refresh

        elif(message['type'] == 'insert_network'):
            _async = False

            # Input Validations
            while (True): # loop just to break from it on first error
                if (data['network_name'] is None or data['network_name'] == ''):
                    error_message = 'INPUT VALIDATION: Name is required.'; break
                if (data['dataset_id'] is None or data['dataset_id'] == ''):
                    error_message = 'INPUT VALIDATION: Dataset is required.'; break
                if (data['input_activation_function'] is None or data['input_activation_function'] == ''):
                    error_message = 'INPUT VALIDATION: Input Activation Function is required.'; break
                if (data['hidden_activation_function'] is None or data['hidden_activation_function'] == ''):
                    error_message = 'INPUT VALIDATION: Hidden Activation Function is required.'; break
                if (data['output_activation_function'] is None or data['output_activation_function'] == ''):
                    error_message = 'INPUT VALIDATION: Output Activation Function is required.'; break
                if (data['loss_function'] is None or data['loss_function'] == ''):
                    error_message = 'INPUT VALIDATION: Loss Function is required.'; break
                if (data['num_inputs'] is None or data['num_inputs'] == ''):
                    error_message = 'INPUT VALIDATION: Number of Inputs is required.'; break
                if (data['num_hidden_layers'] is None or data['num_hidden_layers'] == ''):
                    error_message = 'INPUT VALIDATION: Number of Hidden Layers is required.'; break
                if (data['num_hidden_nodes'] is None or data['num_hidden_nodes'] == ''):
                    error_message = 'INPUT VALIDATION: Number of Hidden Nodes is required.'; break
                if (data['num_output_nodes'] is None or data['num_output_nodes'] == ''):
                    error_message = 'INPUT VALIDATION: Number of Output Nodes is required.'; break
                
                if(not is_numeric(data['dataset_id'])):
                    error_message = 'INPUT VALIDATION: Dataset ID must be numeric.'; break
                if(not is_numeric(data['num_inputs'])):
                    error_message = 'INPUT VALIDATION: Num Inputs must be numeric.'; break
                if(not is_numeric(data['num_hidden_layers'])):
                    error_message = 'INPUT VALIDATION: Num Hidden Layers must be numeric.'; break
                if(not is_numeric(data['num_hidden_nodes'])):
                    error_message = 'INPUT VALIDATION: Num Hidden Nodes must be numeric.'; break
                if(not is_numeric(data['num_output_nodes'])):
                    error_message = 'INPUT VALIDATION: Num Output Nodes must be numeric.'; break
                
                query = 'call sp_insert_network(%s,%s,ARRAY[%s,%s,%s],%s,%s,%s,%s,%s)'
                query_values = [
                    data['network_name'],
                    int(data['dataset_id']),
                    data['input_activation_function'],
                    data['hidden_activation_function'],
                    data['output_activation_function'],
                    data['loss_function'],
                    int(data['num_inputs']),
                    int(data['num_hidden_layers']),
                    int(data['num_hidden_nodes']),
                    int(data['num_output_nodes'])
                ]

                break #end of validations

            if (error_message is not None):
                emit('message_event', {'data' : error_message} )
            elif (query == ""):
                emit('message_event', {'data' : 'Cannot execute empty query'} )
            else:
                db_execute(socketio, query, query_values, _async)
                socket_message({'type':'refresh_network', 'data':''}) # force refresh


        elif(message['type'] == 'train_network'):
            _async = True

            print("")
            print("&& TRAIN NETWORK &&")
            print("Data: %s" % data)

            # Input Validations
            while (True): # loop just to break from it on first error
                if (data['network_id'] is None or data['network_id'] == ''):
                    error_message = 'INPUT VALIDATION: Network ID is required.'; break
                if (data['num_epochs'] is None or data['num_epochs'] == ''):
                    error_message = 'INPUT VALIDATION: Number of Epochs is required.'; break
                if (data['batch_size'] is None or data['batch_size'] == ''):
                    error_message = 'INPUT VALIDATION: Batch Size is required.'; break
                if (data['learning_rate'] is None or data['learning_rate'] == ''):
                    error_message = 'INPUT VALIDATION: Learning Rate is required.'; break

                if (int(data['network_id']) <= 0):
                    error_message = 'INPUT VALIDATION: Network ID must be greater than 0.'; break

                if(not is_numeric(data['network_id'])):
                    error_message = 'INPUT VALIDATION: Network ID must be numeric.'; break
                if(not is_numeric(data['num_epochs'])):
                    error_message = 'INPUT VALIDATION: Num Epochs must be numeric.'; break
                if(not is_numeric(data['batch_size'])):
                    error_message = 'INPUT VALIDATION: Batch Size must be numeric.'; break
                if(not is_numeric(data['learning_rate'])):
                    error_message = 'INPUT VALIDATION: Learning Rate must be numeric.'; break
                
                query = 'call sp_train_network(%s,%s,%s,%s)'
                query_values = [
                    int(data['network_id']),
                    int(data['num_epochs']),
                    int(data['batch_size']),
                    float(data['learning_rate'])
                ]

                break #end of validations

            if (error_message is not None):
                emit('message_event', {'data':error_message} )
            elif (query == ""):
                emit('message_event', {'data':'Cannot execute empty query'} )
            else:
                db_execute(socketio, query, query_values, _async)


        elif(message['type'] == 'set_training_samples'):
            _async = False

            # Input Validations
            while (True): # loop just to break from it on first error
                if (data['network_id'] is None or data['network_id'] == ''):
                    error_message = 'INPUT VALIDATION: Network ID is required.'; break
                if (data['percent_testing'] is None or data['percent_testing'] == ''):
                    error_message = 'INPUT VALIDATION: Percent Test is required.'; break

                if (int(data['network_id']) <= 0):
                    error_message = 'INPUT VALIDATION: Network ID must be greater than 0.'; break
                if (int(data['percent_testing']) < 0 or int(data['percent_testing']) > 100):
                    error_message = 'INPUT VALIDATION: Percent Test Samples must be between 0 and 100.'; break

                query = 'call sp_set_training_samples(%s,%s)'
                query_values = [data['network_id'], data['percent_testing']]

                break #end of validations

            if (error_message is not None):
                emit('message_event', {'data' : error_message} )
            elif (query == ""):
                emit('message_event', {'data' : 'Cannot execute empty query'} )
            else:
                db_execute(socketio, query, query_values, _async)

                
        elif(message['type'] == 'test_network'):
            _async = False

            # Input Validations
            while (True): # loop just to break from it on first error
                if (data['network_id'] is None or data['network_id'] == ''):
                    error_message = 'INPUT VALIDATION: Network ID is required.'; break
                if (data['test_samples_only'] is None or data['test_samples_only'] == ''):
                    error_message = 'INPUT VALIDATION: Training Samples Only is required.'; break
                if (data['show_samples'] is None or data['show_samples'] == ''):
                    error_message = 'INPUT VALIDATION: Show Samples is required.'; break
                    #if (data['num_samples'] is None or data['num_samples'] == ''):
                #    error_message = 'INPUT VALIDATION: Number of Samples is required.'; break
                if (data['num_samples'] == ''):
                    data['num_samples'] = None

                if (int(data['network_id']) <= 0):
                    error_message = 'INPUT VALIDATION: Network ID must be greater than 0.'; break

                if(not is_numeric(data['network_id'])):
                    error_message = 'INPUT VALIDATION: Network ID must be numeric.'; break
                if(data['num_samples'] is not None and not is_numeric(data['num_samples'])):
                    error_message = 'INPUT VALIDATION: Number of Samples must be numeric.'; break

                query = """
                    SELECT test_sample_id,
                            output_num,
                            label_value,
                            output_value,
                            sm_output_value,
                            loss,
                            accuracy,
                            type
                    FROM fx_test_network(%s,%s,%s,%s)
                    ORDER BY type DESC
                """
                query_values = [
                    int(data['network_id']),
                    bool(data['test_samples_only']),
                    bool(data['show_samples']),
                    int(data['num_samples']) if data['num_samples'] is not None else None
                ]

                break #end of validations

            if (error_message is not None):
                emit('message_event', {'data':error_message} )
            elif (query == ""):
                emit('message_event', {'data':'Cannot execute empty query'} )
            else:
                test_list = []

                rows = db_execute(socketio, query, query_values, _async)
                if rows is not None:
                    for row in rows:
                        try:
                            test_list.append({'test_sample_id':row[0],
                                                'output_num':row[1],
                                                'label_value':row[2],
                                                'output_value':row[3],
                                                'sm_output_value':row[4],
                                                'loss':row[5],
                                                'accuracy':row[6],
                                                'type':row[7]})
                        except Exception:
                            pass
                    emit('test_network_event', {'data':test_list})
                else:
                    print('Test Network rows is None')
        
    except KeyError as e:
        print('! Button Event Exception (Key Error): %s' % e)
        emit('message_event', {'data':'Button Input Error (Key Error): %s' % e} )
    except Exception as e:
        print('! Button Event Exception: %s' % e)
        emit('message_event', {'data':'Button Input Error: %s' % e} )


######################
# /mldb HOME
######################
@app.route('/mldb', methods=['GET'])
def mldb_home():
    network_list = []
    dataset_list = []

    network_query = """
        SELECT  nna.id,
                nna.dataset_id,
                nna.activation_functions,
                nna.loss_function,
                nna.name,
                nna.num_hidden_layers,
                nna.layer_sizes,
                d.name AS dataset_name
        FROM neural_network_architecture nna
        JOIN dataset d ON nna.dataset_id = d.id
        ORDER BY id DESC
    """

    dataset_query = """
        SELECT  id,
                name,
                normalize_features,
                normalize_labels
        FROM dataset
        ORDER BY id
    """

    try:
        _async = False
        query_values = None

        # Get network data
        rows = db_execute(socketio, network_query, query_values, _async)
        if rows is not None:
            for row in rows:
                try:
                    network_activation_functions = ['' if len(row[2]) < 1 else row[2][0],
                                                    '' if len(row[2]) < 2 else row[2][1],
                                                    '' if len(row[2]) < 3 else row[2][2]]
                    layer_sizes_arr = ['' if len(row[6]) < 1 else row[6][0],
                                            '' if len(row[6]) < 2 else row[6][1],
                                            '' if len(row[6]) < 3 else row[6][2]]
                    layer_sizes_str = 'IN:%d, HDN:%dx%d, OUT:%d' % (layer_sizes_arr[0],row[5],layer_sizes_arr[1],layer_sizes_arr[2])                                        
                    network_list.append({'id':row[0],
                                        'name':row[4],
                                        'activation_functions':network_activation_functions,
                                        'loss_function':row[3],
                                        'layer_sizes':layer_sizes_str,
                                        'dataset_name':row[7]})
                except Exception:
                    pass

        # Get dataset data
        rows = db_execute(socketio, dataset_query, query_values, _async)
        if rows is not None:
            for row in rows:
                dataset_value = "ID:%d (%s)" % (row[0], row[1])
                dataset_list.append({"id":row[0], "name":row[1], "display_name":dataset_value})

    except psycopg2.DatabaseError as e:
        print('DB Error: %s' % e)
        socketio.emit('message_event', {'data':"DB Error: %s" % e}, namespace='/mldbns')
        #sys.exit(1)
    except Exception as e:
        print('Other DB Error: %s' % e)
        socketio.emit('message_event', {'data':"Exception: %s" % e}, namespace='/mldbns')
 
    activation_function_list = [{'name':'LINEAR', 'display_name':'Linear'},
                                {'name':'SIGMOID', 'display_name':'Sigmoid'},
                                {'name':'TANH', 'display_name':'Tanh'},
                                {'name':'RELU', 'display_name':'ReLU'}]
    loss_function_list = [{'name':'MSE', 'display_name':'Mean Squared Error'},
                            {'name':'CROSS_ENTROPY', 'display_name':'Cross Entropy'}]
    normalization_list = [{'name':'', 'display_name':'None'},
                            {'name':'MINMAX', 'display_name':'Min-Max'},
                            {'name':'ZSCORE', 'display_name':'Z-Score'}]

    return render_template('mldb_home.html',
                            network_list=network_list,
                            activation_function_list=activation_function_list,
                            dataset_list=dataset_list,
                            loss_function_list=loss_function_list,
                            normalization_list=normalization_list
                            )


###############
# Index
###############

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

###############
# 404 Page
###############

@app.errorhandler(404)
def e404(e):
    return render_template('404.html'), 404

###############
# Run with SSL
###############

if __name__ == '__main__':  # Run App
    #pkey    = '/etc/apache2/ssl/server.key'
    #cert    = '/etc/apache2/ssl/server.crt'
    #context = SSL.Context(SSL.SSLv23_METHOD)
    #context.use_privatekey_file(pkey)
    #context.use_certificate_file(cert)
    #socketio.run(app)
    #app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=(cert,pkey))
    socketio.run(app, 
                    debug=True,
                    host='0.0.0.0',
                    port=5000 
                    #,certfile=cert,
                    #keyfile=pkey
                    )
    #socketio.run(app, debug=False, host='0.0.0.0', ssl_context=(cert,pkey))
    
