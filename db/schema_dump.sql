--
-- PostgreSQL database dump
--

-- Dumped from database version 12.2
-- Dumped by pg_dump version 12.2

-- Started on 2020-04-16 10:36:41

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 11 (class 2615 OID 16396)
-- Name: uofl; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA uofl;


--
-- TOC entry 1 (class 3079 OID 16397)
-- Name: adminpack; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS adminpack WITH SCHEMA pg_catalog;


--
-- TOC entry 3117 (class 0 OID 0)
-- Dependencies: 1
-- Name: EXTENSION adminpack; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION adminpack IS 'administrative functions for PostgreSQL';


--
-- TOC entry 4 (class 3079 OID 16406)
-- Name: hstore; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS hstore WITH SCHEMA public;


--
-- TOC entry 3118 (class 0 OID 0)
-- Dependencies: 4
-- Name: EXTENSION hstore; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION hstore IS 'data type for storing sets of (key, value) pairs';


--
-- TOC entry 3 (class 3079 OID 16531)
-- Name: tablefunc; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS tablefunc WITH SCHEMA public;


--
-- TOC entry 3119 (class 0 OID 0)
-- Dependencies: 3
-- Name: EXTENSION tablefunc; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION tablefunc IS 'functions that manipulate whole tables, including crosstab';


--
-- TOC entry 334 (class 1255 OID 16552)
-- Name: fn_identity(double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fn_identity(x double precision) RETURNS double precision
    LANGUAGE sql IMMUTABLE STRICT
    AS $$
 SELECT x
$$;


--
-- TOC entry 335 (class 1255 OID 16553)
-- Name: fn_isnumeric(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fn_isnumeric(val text) RETURNS boolean
    LANGUAGE plpgsql
    AS $$
DECLARE n int;
BEGIN
	n := val::numeric;
	RETURN true;
	EXCEPTION WHEN invalid_text_representation THEN
	RETURN false;
END;
$$;


--
-- TOC entry 336 (class 1255 OID 16554)
-- Name: fn_mse(double precision, double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fn_mse(expected double precision, observed double precision) RETURNS double precision
    LANGUAGE sql IMMUTABLE STRICT
    AS $$
	SELECT (0.5) * power(expected - observed, 2)
$$;


--
-- TOC entry 337 (class 1255 OID 16555)
-- Name: fn_relu(double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fn_relu(x double precision) RETURNS double precision
    LANGUAGE sql IMMUTABLE STRICT
    AS $$
 SELECT MAX(n)
 FROM (SELECT x
	  	UNION ALL
	  	VALUES(0.0)) t(n)
$$;


--
-- TOC entry 338 (class 1255 OID 16556)
-- Name: fn_relu_prime(double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fn_relu_prime(x double precision) RETURNS double precision
    LANGUAGE sql IMMUTABLE STRICT
    AS $$
 SELECT CAST(CASE WHEN x > 0 
 				THEN 1.0
			ELSE 0.0
		END AS float)
$$;


--
-- TOC entry 339 (class 1255 OID 16557)
-- Name: fn_sigmoid(double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fn_sigmoid(x double precision) RETURNS double precision
    LANGUAGE sql IMMUTABLE STRICT
    AS $$
	-- o = 1 / (1 + exp(-x))
 	SELECT 1.0 / (1 + power(2.71828182845904523536028747, -1.0*x))
$$;


--
-- TOC entry 340 (class 1255 OID 16558)
-- Name: fn_sigmoid_prime(double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fn_sigmoid_prime(x double precision) RETURNS double precision
    LANGUAGE sql IMMUTABLE STRICT
    AS $$
	-- o = 1 / (1 + exp(-x))
	-- o' = o(1 - o)
 	SELECT (1.0 / (1 + power(2.71828182845904523536028747, -1.0*x)))
 			* (1 - (1.0 / (1 + power(2.71828182845904523536028747, -1.0*x))))
$$;


--
-- TOC entry 341 (class 1255 OID 16559)
-- Name: fn_tanh(double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fn_tanh(x double precision) RETURNS double precision
    LANGUAGE sql IMMUTABLE STRICT
    AS $$
 SELECT (power(2.71828182845904523536028747, x) - power(2.71828182845904523536028747, -1.0*x))
 		/ (power(2.71828182845904523536028747, x) + power(2.71828182845904523536028747, -1.0*x))
$$;


--
-- TOC entry 342 (class 1255 OID 16560)
-- Name: fn_tanh_prime(double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fn_tanh_prime(x double precision) RETURNS double precision
    LANGUAGE sql IMMUTABLE STRICT
    AS $$

	-- 1 - tanh(x)^2
	
	SELECT 1 - power((power(2.71828182845904523536028747, x) - power(2.71828182845904523536028747, -1.0*x))
					 	/ (power(2.71828182845904523536028747, x) + power(2.71828182845904523536028747, -1.0*x))
					 , 2)
	
	--SELECT (power(2.71828182845904523536028747, x) - power(2.71828182845904523536028747, -1.0*x))
 	--	/ (power(2.71828182845904523536028747, x) + power(2.71828182845904523536028747, -1.0*x))
$$;


--
-- TOC entry 343 (class 1255 OID 16561)
-- Name: fx_feature_classify(integer, double precision[]); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fx_feature_classify(v_network_id integer, v_features double precision[]) RETURNS TABLE(o_result_num integer, o_result double precision, o_sm_result double precision, o_loss double precision)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	lv_dataset_id int;
	lv_input_activation_fn text;
	lv_hidden_activation_fn text;
	lv_output_activation_fn text;
	lv_normalize_features boolean;
	lv_loss_fn text;
	lv_num_layers integer;
	--lv_num_samples int;
	lv_layer_iter RECORD;
BEGIN

	---------------------------
	-- Initialization
	---------------------------
	
	-- Get network information
	SELECT 
		dataset_id,
		activation_functions[1],
		activation_functions[2],
		activation_functions[3],
		loss_function
	INTO
		lv_dataset_id,
		lv_input_activation_fn,
		lv_hidden_activation_fn,
		lv_output_activation_fn,
		lv_loss_fn
	FROM neural_network_architecture
	WHERE id = v_network_id;

	-- Get dataset information
	SELECT
		normalize_features
	INTO
		lv_normalize_features
	FROM dataset
	WHERE id = lv_dataset_id;

	-- Calculate layers in network
	SELECT MAX(layer_num)
	INTO lv_num_layers
	FROM neural_network
	WHERE network_id = v_network_id;
	
	-- Declare temp tables
	CREATE TEMP TABLE tmp_layer_state
	(
		layer_num int,
		node_num int,
		result double precision,   -- result of matrix multiplication before activation fn
		activity double precision   -- result after activation function
	);

	---------------------------
	-- Forward Propagation
	---------------------------
	
	-- Insert input node weights into table
	WITH 
	statistics_cte 
	AS (
		SELECT feature_num, mean, std
		FROM dataset ds,
			UNNEST(feature_mean,feature_std) WITH ORDINALITY AS t(mean, std, feature_num)
		WHERE ds.id = lv_dataset_id),
	feature_norm_cte 
	AS (SELECT sf.feature_num,
				CASE WHEN lv_normalize_features = true
					THEN
						(sf.value - COALESCE(sc.mean,0)) / CASE WHEN sc.std = 0 OR sc.std IS NULL THEN 1 ELSE sc.std END
					ELSE sf.value
				END AS feature_value
		FROM UNNEST(v_features) WITH ORDINALITY AS sf(value, feature_num)
		LEFT JOIN statistics_cte sc ON sf.feature_num = sc.feature_num),
	feature_all_cte 
	AS (SELECT feature_num,
				feature_value
		FROM feature_norm_cte
	   
	   	UNION ALL
	   	-- Augment inputs with 1-valued features to match with bias
	   	SELECT MAX(feature_num) + 1,
				1   -- feature_value
		FROM feature_norm_cte),
	layer_cte
	AS (
		SELECT 1 AS layer_num,
				nn.weight_num, 
				-- (x - mean) / std   for input normalization
				SUM(fa.feature_value * nn.weight) AS result
		FROM feature_all_cte fa
		JOIN vw_neural_network nn ON nn.node_num = fa.feature_num   -- line up inputs and weights
		WHERE nn.network_id = v_network_id   -- network id parameter
			AND nn.layer_num = 1   -- input layer
		GROUP BY nn.weight_num
	)
	INSERT INTO tmp_layer_state (layer_num, node_num, result, activity)
	SELECT layer_num,
			weight_num,
			result,
			CASE WHEN lv_input_activation_fn = 'LINEAR'
					THEN result
				 WHEN lv_input_activation_fn = 'TANH'
					THEN fn_tanh(result)
				 WHEN lv_input_activation_fn = 'RELU'
					THEN fn_relu(result)
				 WHEN lv_input_activation_fn = 'SIGMOID'
					THEN fn_sigmoid(result)
				ELSE result
			END   -- determine activation function for input layer
	FROM layer_cte;
	
	
	-- Loop across layers for forward propagation
	FOR lv_layer_iter IN
		SELECT generate_series(2, lv_num_layers-1) AS layer_num
	LOOP
		WITH
		prev_layer_cte
		AS (SELECT layer_num, node_num, activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1

			UNION ALL

			-- Append 1-valued inputs to match with bias
			SELECT layer_num,
					MAX(node_num) + 1,   -- extra node in next layer
					1    -- activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1
			GROUP BY layer_num),
		layer_cte AS (	
			SELECT lv_layer_iter.layer_num,
					nn.weight_num, 
					-- z = a * W
					SUM(ls.activity * nn.weight) AS result
			FROM prev_layer_cte ls
			JOIN vw_neural_network nn ON nn.node_num = ls.node_num
			WHERE nn.network_id = v_network_id 
				AND nn.layer_num = lv_layer_iter.layer_num   -- correct network layer
			GROUP BY nn.weight_num)
		-- insert result of matrix multiplication in table for this layer
		INSERT INTO tmp_layer_state (layer_num, node_num, result, activity)
		SELECT layer_num,
				weight_num,
				result,
				CASE WHEN lv_hidden_activation_fn = 'LINEAR'
						THEN result
					 WHEN lv_hidden_activation_fn = 'TANH'
						THEN fn_tanh(result)
					 WHEN lv_hidden_activation_fn = 'RELU'
						THEN fn_relu(result)
					 WHEN lv_hidden_activation_fn = 'SIGMOID'
						THEN fn_sigmoid(result)
					 ELSE result
				END   -- determine activation function for hidden layer(s)
		FROM layer_cte;
			
	END LOOP;	-- end hidden layer loop
	
	
	-- Calculate last layer and network outputs
	WITH
	prev_layer_cte
	AS (SELECT layer_num, node_num, activity
		FROM tmp_layer_state ls
		WHERE ls.layer_num = lv_num_layers - 1

		UNION ALL
		-- Append 1-valued inputs to match with bias
		SELECT layer_num,
				MAX(node_num) + 1,   -- extra node in next layer
				1    -- activity
		FROM tmp_layer_state ls
		WHERE ls.layer_num = lv_num_layers - 1
		GROUP BY layer_num),
	layer_cte 
	AS (SELECT lv_num_layers AS layer_num,
				nn.weight_num,
				-- z = a * W
				SUM(ls.activity * nn.weight) AS result
		FROM prev_layer_cte ls
		JOIN vw_neural_network nn ON nn.node_num = ls.node_num
		WHERE nn.network_id = v_network_id 
			AND nn.layer_num = lv_num_layers   -- correct network layer
		GROUP BY nn.weight_num)
	-- insert result of matrix multiplication in table for this layer
	INSERT INTO tmp_layer_state (layer_num, node_num, result, activity)
	SELECT layer_num,
			weight_num,
			result,
			CASE WHEN lv_output_activation_fn = 'LINEAR'
					THEN result
				 WHEN lv_output_activation_fn = 'TANH'
					THEN fn_tanh(result)
				 WHEN lv_output_activation_fn = 'RELU'
					THEN fn_relu(result)
				 WHEN lv_output_activation_fn = 'SIGMOID'
					THEN fn_sigmoid(result)
				 ELSE result
			END   -- determine activation function for output layer(s)
	FROM layer_cte;
	

	RETURN QUERY
		--o_result_num integer, o_result float, o_sm_result float, o_loss float) 
		
		-- Choose highest valued activity
		WITH 
		softmax_cte
		AS (SELECT node_num,
					activity,
		   			EXP(activity) / (SUM(EXP(activity)) OVER ()) AS sm_result,
					ROW_NUMBER() OVER (PARTITION BY layer_num ORDER BY activity DESC) AS row_num
			FROM tmp_layer_state ls
			WHERE layer_num = lv_num_layers)
		SELECT node_num,
				activity,
				sm_result,
				-1*LOG(sm_result)
		FROM softmax_cte
		WHERE row_num = 1   -- Get active node_num corresponding to class number
		ORDER BY node_num;
		
	
	-- Drop temp tables
	DROP TABLE tmp_layer_state;	
	
END;
$$;


--
-- TOC entry 344 (class 1255 OID 16563)
-- Name: fx_feature_regression(integer, double precision[]); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fx_feature_regression(v_network_id integer, v_features double precision[]) RETURNS TABLE(o_result_num integer, o_result double precision)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	/*
	v_network_id integer,
	v_features double precision[]
	*/
	lv_dataset_id int;
	lv_input_activation_fn text;
	lv_hidden_activation_fn text;
	lv_output_activation_fn text;
	lv_loss_fn text;
	lv_num_layers integer;
	--lv_num_samples int;
	lv_layer_iter RECORD;
BEGIN

	---------------------------
	-- Initialization
	---------------------------
	
	-- Get network information
	SELECT 
		dataset_id,
		activation_functions[1],
		activation_functions[2],
		activation_functions[3],
		loss_function
	INTO
		lv_dataset_id,
		lv_input_activation_fn,
		lv_hidden_activation_fn,
		lv_output_activation_fn,
		lv_loss_fn
	FROM neural_network_architecture
	WHERE id = v_network_id;
	
	-- Calculate layers in network
	lv_num_layers := MAX(layer_num)
	FROM neural_network
	WHERE network_id = v_network_id;
	
	-- Declare temp tables
	CREATE TEMP TABLE tmp_layer_state
	(
		layer_num int,
		node_num int,
		result double precision,   -- result of matrix multiplication before activation fn
		activity double precision   -- result after activation function
	);

	---------------------------
	-- Forward Propagation
	---------------------------
	
	-- Insert input node weights into table
	WITH 
	statistics_cte 
	AS (
		SELECT feature_num, mean, std
		FROM dataset ds,
			UNNEST(feature_mean,feature_std) WITH ORDINALITY AS t(mean, std, feature_num)
		WHERE ds.id = lv_dataset_id),
	feature_norm_cte 
	AS (SELECT sf.feature_num,
				(sf.value - COALESCE(sc.mean,0)) / CASE WHEN sc.std = 0 OR sc.std IS NULL THEN 1 ELSE sc.std END AS feature_value
		FROM UNNEST(v_features) WITH ORDINALITY AS sf(value, feature_num)
		LEFT JOIN statistics_cte sc ON sf.feature_num = sc.feature_num),
	feature_all_cte 
	AS (SELECT feature_num,
				feature_value
		FROM feature_norm_cte
	   
	   	UNION ALL
		-- Augment inputs with 1-valued features to match with bias
	   	SELECT MAX(feature_num) + 1,
				1   -- feature_value
		FROM feature_norm_cte),
	layer_cte
	AS (
		SELECT 1 AS layer_num,
				nn.weight_num, 
				-- (x - mean) / std   for input normalization
				SUM(fa.feature_value * nn.weight) AS result
		FROM feature_all_cte fa
		JOIN vw_neural_network nn ON nn.node_num = fa.feature_num   -- line up inputs and weights
		WHERE nn.network_id = v_network_id   -- network id parameter
			AND nn.layer_num = 1   -- input layer
		GROUP BY nn.weight_num
	)
	-- insert result of matrix multiplication in table for this layer
	INSERT INTO tmp_layer_state (layer_num, node_num, result, activity)
	SELECT layer_num,
			weight_num,
			result,
			CASE WHEN lv_input_activation_fn = 'LINEAR'
					THEN result
				 WHEN lv_input_activation_fn = 'TANH'
					THEN fn_tanh(result)
				 WHEN lv_input_activation_fn = 'RELU'
					THEN fn_relu(result)
				 WHEN lv_input_activation_fn = 'SIGMOID'
					THEN fn_sigmoid(result)
				ELSE result
			END   -- determine activation function for input layer
	FROM layer_cte;
	
	
	-- Loop across layers for forward propagation
	FOR lv_layer_iter IN
		SELECT generate_series(2, lv_num_layers-1) AS layer_num
	LOOP
		WITH 
		prev_layer_cte
		AS (SELECT layer_num, node_num, activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1

			UNION ALL

			-- Append 1-valued inputs to match with bias
			SELECT layer_num,
					MAX(node_num) + 1,   -- extra node in next layer
					1    -- activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1
			GROUP BY layer_num),
		layer_cte AS (	
			SELECT lv_layer_iter.layer_num,
					nn.weight_num, 
					-- z = a * W
					SUM(ls.activity * nn.weight) AS result
			FROM prev_layer_cte ls
			JOIN vw_neural_network nn ON nn.node_num = ls.node_num
			WHERE nn.network_id = v_network_id 
				AND nn.layer_num = lv_layer_iter.layer_num   -- correct network layer
			GROUP BY nn.weight_num)
		-- insert result of matrix multiplication in table for this layer
		INSERT INTO tmp_layer_state  (layer_num, node_num, result, activity)
		SELECT layer_num,
				weight_num,
				result,
				CASE WHEN lv_hidden_activation_fn = 'LINEAR'
						THEN result
					 WHEN lv_hidden_activation_fn = 'TANH'
						THEN fn_tanh(result)
					 WHEN lv_hidden_activation_fn = 'RELU'
						THEN fn_relu(result)
					 WHEN lv_hidden_activation_fn = 'SIGMOID'
						THEN fn_sigmoid(result)
					 ELSE result
				END   -- determine activation function for hidden layer(s)
		FROM layer_cte;
			
	END LOOP;	-- end hidden layer loop
	
	
	-- Calculate last layer and network outputs
	WITH
	prev_layer_cte
	AS (SELECT layer_num, node_num, activity
		FROM tmp_layer_state ls
		WHERE ls.layer_num = lv_num_layers - 1

		UNION ALL
		-- Append 1-valued inputs to match with bias
		SELECT layer_num,
				MAX(node_num) + 1,   -- extra node in next layer
				1    -- activity
		FROM tmp_layer_state ls
		WHERE ls.layer_num = lv_num_layers - 1
		GROUP BY layer_num),
	layer_cte 
	AS (SELECT lv_num_layers AS layer_num,
				nn.weight_num,
				-- z = a * W
				SUM(ls.activity * nn.weight) AS result
		FROM prev_layer_cte ls
		JOIN vw_neural_network nn ON nn.node_num = ls.node_num
		WHERE nn.network_id = v_network_id 
			AND nn.layer_num = lv_num_layers   -- correct network layer
		GROUP BY nn.weight_num)
	-- insert result of matrix multiplication in table for this layer
	INSERT INTO tmp_layer_state (layer_num, node_num, result, activity)
	SELECT layer_num,
			weight_num,
			result,
			CASE WHEN lv_output_activation_fn = 'LINEAR'
					THEN result
				 WHEN lv_output_activation_fn = 'TANH'
					THEN fn_tanh(result)
				 WHEN lv_output_activation_fn = 'RELU'
					THEN fn_relu(result)
				 WHEN lv_output_activation_fn = 'SIGMOID'
					THEN fn_sigmoid(result)
				 ELSE result
			END   -- determine activation function for output layer(s)
	FROM layer_cte;
	

	RETURN QUERY
		WITH
		statistics_cte 
		AS (
			SELECT label_num,
					label_min_value,
					label_max_value
			FROM dataset ds,
				UNNEST(label_min, label_max) WITH ORDINALITY AS l(label_min_value, label_max_value, label_num)
			WHERE ds.id = lv_dataset_id)

		SELECT CAST(sc.label_num AS int) AS o_result_num,
				ls.activity * (sc.label_max_value - sc.label_min_value) + sc.label_min_value AS o_result
		FROM tmp_layer_state ls
		JOIN statistics_cte sc ON ls.node_num = sc.label_num
		WHERE layer_num = lv_num_layers;

	
	-- Drop temp tables
	DROP TABLE tmp_layer_state;	
	
END;
$$;


--
-- TOC entry 359 (class 1255 OID 35893)
-- Name: fx_forward_network(integer, double precision[]); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fx_forward_network(v_network_id integer, v_features double precision[]) RETURNS TABLE(output_num integer, output double precision, sm_output double precision)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	/*
	v_network_id integer,
	v_features double precision[]
	*/
	lv_dataset_id int;
	lv_input_activation_fn text;
	lv_hidden_activation_fn text;
	lv_output_activation_fn text;
	lv_normalize_features text;
	lv_loss_fn text;
	lv_num_layers integer;
	lv_layer_iter RECORD;
BEGIN

	---------------------------
	-- Initialization
	---------------------------
	
	-- Get network information
	SELECT 
		dataset_id,
		activation_functions[1],
		activation_functions[2],
		activation_functions[3],
		loss_function
	INTO
		lv_dataset_id,
		lv_input_activation_fn,
		lv_hidden_activation_fn,
		lv_output_activation_fn,
		lv_loss_fn
	FROM neural_network_architecture
	WHERE id = v_network_id;
	
	-- Calculate layers in network
	SELECT MAX(layer_num)
	INTO lv_num_layers
	FROM neural_network
	WHERE network_id = v_network_id;
	
   	-- Get dataset information
	SELECT
		normalize_features
	INTO
		lv_normalize_features
	FROM dataset
	WHERE id = lv_dataset_id;
	
	-- Declare temp tables
	CREATE TEMP TABLE tmp_layer_state
	(
		layer_num int,
		node_num int,
		result double precision,   -- result of matrix multiplication before activation fn
		activity double precision   -- result after activation function
	);

	---------------------------
	-- Forward Propagation
	---------------------------
	
	-- Insert input node weights into table
	WITH 
	statistics_cte
	AS (SELECT f.feature_num,
				 f.feature_mean_value,
				 f.feature_std_value,
				 f.feature_max_value,
				 f.feature_min_value
		FROM dataset d,
			UNNEST(feature_mean, feature_std, feature_max, feature_min)
				WITH ORDINALITY f(feature_mean_value, feature_std_value, feature_max_value, feature_min_value, feature_num)
	   WHERE d.id = lv_dataset_id),
	feature_norm_cte 
	AS (SELECT f.feature_num,
				CASE WHEN lv_normalize_features = 'ZSCORE'
						THEN (f.feature_value - COALESCE(s.feature_mean_value,0)) / CASE WHEN (s.feature_std_value = 0 OR s.feature_std_value IS NULL) THEN 1 ELSE s.feature_std_value END   -- normalized value
					WHEN lv_normalize_features = 'MINMAX'
						THEN (f.feature_value - COALESCE(s.feature_max_value,0)) / (s.feature_max_value - s.feature_min_value)   -- normalized value
				END AS feature_value
		FROM UNNEST(v_features) WITH ORDINALITY AS f(feature_value, feature_num)
		LEFT JOIN statistics_cte s ON f.feature_num = s.feature_num),
	feature_all_cte 
	AS (SELECT feature_num,
				feature_value
		FROM feature_norm_cte
	   
	   	UNION ALL
		-- Augment inputs with 1-valued features to match with bias
	   	SELECT MAX(feature_num) + 1,
				1   -- feature_value
		FROM feature_norm_cte),
	layer_cte
	AS (
		SELECT 1 AS layer_num,
				nn.weight_num, 
				-- (x - mean) / std   for input normalization
				SUM(fa.feature_value * nn.weight) AS result
		FROM feature_all_cte fa
		JOIN vw_neural_network nn ON nn.node_num = fa.feature_num   -- line up inputs and weights
		WHERE nn.network_id = v_network_id   -- network id parameter
			AND nn.layer_num = 1   -- input layer
		GROUP BY nn.weight_num
	)
	-- insert result of matrix multiplication in table for this layer
	INSERT INTO tmp_layer_state (layer_num, node_num, result, activity)
	SELECT layer_num,
			weight_num,
			result,
			CASE WHEN lv_input_activation_fn = 'LINEAR'
					THEN result
				 WHEN lv_input_activation_fn = 'TANH'
					THEN fn_tanh(result)
				 WHEN lv_input_activation_fn = 'RELU'
					THEN fn_relu(result)
				 WHEN lv_input_activation_fn = 'SIGMOID'
					THEN fn_sigmoid(result)
				ELSE result
			END   -- determine activation function for input layer
	FROM layer_cte;
	
	
	-- Loop across layers for forward propagation
	FOR lv_layer_iter IN
		SELECT generate_series(2, lv_num_layers-1) AS layer_num
	LOOP
		WITH 
		prev_layer_cte
		AS (SELECT layer_num, node_num, activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1

			UNION ALL

			-- Append 1-valued inputs to match with bias
			SELECT layer_num,
					MAX(node_num) + 1,   -- extra node in next layer
					1    -- activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1
			GROUP BY layer_num),
		layer_cte AS (	
			SELECT lv_layer_iter.layer_num,
					nn.weight_num, 
					-- z = a * W
					SUM(ls.activity * nn.weight) AS result
			FROM prev_layer_cte ls
			JOIN vw_neural_network nn ON nn.node_num = ls.node_num
			WHERE nn.network_id = v_network_id 
				AND nn.layer_num = lv_layer_iter.layer_num   -- correct network layer
			GROUP BY nn.weight_num)
		-- insert result of matrix multiplication in table for this layer
		INSERT INTO tmp_layer_state  (layer_num, node_num, result, activity)
		SELECT layer_num,
				weight_num,
				result,
				CASE WHEN lv_hidden_activation_fn = 'LINEAR'
						THEN result
					 WHEN lv_hidden_activation_fn = 'TANH'
						THEN fn_tanh(result)
					 WHEN lv_hidden_activation_fn = 'RELU'
						THEN fn_relu(result)
					 WHEN lv_hidden_activation_fn = 'SIGMOID'
						THEN fn_sigmoid(result)
					 ELSE result
				END   -- determine activation function for hidden layer(s)
		FROM layer_cte;
			
	END LOOP;	-- end hidden layer loop
	
	
	-- Calculate last layer and network outputs
	WITH
	prev_layer_cte
	AS (SELECT layer_num, node_num, activity
		FROM tmp_layer_state ls
		WHERE ls.layer_num = lv_num_layers - 1

		UNION ALL
		-- Append 1-valued inputs to match with bias
		SELECT layer_num,
				MAX(node_num) + 1,   -- extra node in next layer
				1    -- activity
		FROM tmp_layer_state ls
		WHERE ls.layer_num = lv_num_layers - 1
		GROUP BY layer_num),
	layer_cte 
	AS (SELECT lv_num_layers AS layer_num,
				nn.weight_num,
				-- z = a * W
				SUM(ls.activity * nn.weight) AS result
		FROM prev_layer_cte ls
		JOIN vw_neural_network nn ON nn.node_num = ls.node_num
		WHERE nn.network_id = v_network_id 
			AND nn.layer_num = lv_num_layers   -- correct network layer
		GROUP BY nn.weight_num)
	-- insert result of matrix multiplication in table for this layer
	INSERT INTO tmp_layer_state (layer_num, node_num, result, activity)
	SELECT layer_num,
			weight_num,
			result,
			CASE WHEN lv_output_activation_fn = 'LINEAR'
					THEN result
				 WHEN lv_output_activation_fn = 'TANH'
					THEN fn_tanh(result)
				 WHEN lv_output_activation_fn = 'RELU'
					THEN fn_relu(result)
				 WHEN lv_output_activation_fn = 'SIGMOID'
					THEN fn_sigmoid(result)
				 ELSE result
			END   -- determine activation function for output layer(s)
	FROM layer_cte;
	
	
	IF lv_loss_fn = 'MSE'
	THEN
		RETURN QUERY
			WITH
			statistics_cte 
			AS (
				SELECT label_num,
						label_min_value,
						label_max_value
				FROM dataset ds,
					UNNEST(label_min, label_max) WITH ORDINALITY AS l(label_min_value, label_max_value, label_num)
				WHERE ds.id = lv_dataset_id),
			output_cte
			AS (SELECT CAST(sc.label_num AS int) AS _output_num,
						ls.activity * (sc.label_max_value - sc.label_min_value) + sc.label_min_value AS _output,
						ls.activity
						--(v_label - sc.label_min_value) / (sc.label_max_value - sc.label_min_value) AS label_norm  --* (sc.label_max_value - sc.label_min_value) + sc.label_min_value,
				FROM tmp_layer_state ls
				JOIN statistics_cte sc ON ls.node_num = sc.label_num
				WHERE layer_num = lv_num_layers)
			SELECT _output_num,
					_output,
					_output,
					-- Loss = (1/2) (y - _y)^2
					POWER(label_norm - activity, 2) / 2
			FROM output_cte;
			
			/*SELECT CAST(sc.label_num AS int),
					ls.activity * (sc.label_max_value - sc.label_min_value) + sc.label_min_value,
					NULL::float,
					v_label
			FROM tmp_layer_state ls
			JOIN statistics_cte sc ON ls.node_num = sc.label_num
			WHERE layer_num = lv_num_layers;*/
			
	ELSIF lv_loss_fn = 'CROSS_ENTROPY'
	THEN
		RETURN QUERY
			--o_result_num integer, o_result float, o_sm_result float, o_loss float) 

			-- Choose highest valued activity
			WITH 
			softmax_cte
			AS (SELECT node_num,
						activity,
						EXP(activity) / (SUM(EXP(activity)) OVER ()) AS sm_result,
						ROW_NUMBER() OVER (PARTITION BY layer_num ORDER BY activity DESC) AS row_num
				FROM tmp_layer_state ls
				WHERE layer_num = lv_num_layers)
			SELECT node_num,
					activity,
					sm_result
			FROM softmax_cte
			WHERE row_num = 1   -- Get active node_num corresponding to class number
			ORDER BY node_num;
	ELSE
		RETURN QUERY
			SELECT NULL,NULL,NULL,NULL;
	END IF;
	
	-- Drop temp tables
	DROP TABLE tmp_layer_state;	
	
END;
$$;


--
-- TOC entry 345 (class 1255 OID 16565)
-- Name: fx_forward_network(integer, double precision[], double precision); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fx_forward_network(v_network_id integer, v_features double precision[], v_label double precision) RETURNS TABLE(output_num integer, output double precision, sm_output double precision, loss double precision)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	/*
	v_network_id integer,
	v_features double precision[]
	*/
	lv_dataset_id int;
	lv_input_activation_fn text;
	lv_hidden_activation_fn text;
	lv_output_activation_fn text;
	lv_normalize_features boolean;
	lv_loss_fn text;
	lv_num_layers integer;
	lv_layer_iter RECORD;
BEGIN

	---------------------------
	-- Initialization
	---------------------------
	
	-- Get network information
	SELECT 
		dataset_id,
		activation_functions[1],
		activation_functions[2],
		activation_functions[3],
		loss_function
	INTO
		lv_dataset_id,
		lv_input_activation_fn,
		lv_hidden_activation_fn,
		lv_output_activation_fn,
		lv_loss_fn
	FROM neural_network_architecture
	WHERE id = v_network_id;
	
	-- Calculate layers in network
	SELECT MAX(layer_num)
	INTO lv_num_layers
	FROM neural_network
	WHERE network_id = v_network_id;
	
   	-- Get dataset information
	SELECT
		normalize_features
	INTO
		lv_normalize_features
	FROM dataset
	WHERE id = lv_dataset_id;
	
	-- Declare temp tables
	CREATE TEMP TABLE tmp_layer_state
	(
		layer_num int,
		node_num int,
		result double precision,   -- result of matrix multiplication before activation fn
		activity double precision   -- result after activation function
	);

	---------------------------
	-- Forward Propagation
	---------------------------
	
	-- Insert input node weights into table
	WITH 
	statistics_cte 
	AS (
		SELECT feature_num, mean, std
		FROM dataset ds,
			UNNEST(feature_mean,feature_std) WITH ORDINALITY AS t(mean, std, feature_num)
		WHERE ds.id = lv_dataset_id),
	feature_norm_cte 
	AS (SELECT sf.feature_num,
				--(sf.value - COALESCE(sc.mean,0)) / CASE WHEN sc.std = 0 OR sc.std IS NULL THEN 1 ELSE sc.std END AS feature_value
				CASE WHEN lv_normalize_features = true
					THEN
						(sf.value - COALESCE(sc.mean,0)) / CASE WHEN sc.std = 0 OR sc.std IS NULL THEN 1 ELSE sc.std END
					ELSE sf.value
				END AS feature_value
		FROM UNNEST(v_features) WITH ORDINALITY AS sf(value, feature_num)
		LEFT JOIN statistics_cte sc ON sf.feature_num = sc.feature_num),
	feature_all_cte 
	AS (SELECT feature_num,
				feature_value
		FROM feature_norm_cte
	   
	   	UNION ALL
		-- Augment inputs with 1-valued features to match with bias
	   	SELECT MAX(feature_num) + 1,
				1   -- feature_value
		FROM feature_norm_cte),
	layer_cte
	AS (
		SELECT 1 AS layer_num,
				nn.weight_num, 
				-- (x - mean) / std   for input normalization
				SUM(fa.feature_value * nn.weight) AS result
		FROM feature_all_cte fa
		JOIN vw_neural_network nn ON nn.node_num = fa.feature_num   -- line up inputs and weights
		WHERE nn.network_id = v_network_id   -- network id parameter
			AND nn.layer_num = 1   -- input layer
		GROUP BY nn.weight_num
	)
	-- insert result of matrix multiplication in table for this layer
	INSERT INTO tmp_layer_state (layer_num, node_num, result, activity)
	SELECT layer_num,
			weight_num,
			result,
			CASE WHEN lv_input_activation_fn = 'LINEAR'
					THEN result
				 WHEN lv_input_activation_fn = 'TANH'
					THEN fn_tanh(result)
				 WHEN lv_input_activation_fn = 'RELU'
					THEN fn_relu(result)
				 WHEN lv_input_activation_fn = 'SIGMOID'
					THEN fn_sigmoid(result)
				ELSE result
			END   -- determine activation function for input layer
	FROM layer_cte;
	
	
	-- Loop across layers for forward propagation
	FOR lv_layer_iter IN
		SELECT generate_series(2, lv_num_layers-1) AS layer_num
	LOOP
		WITH 
		prev_layer_cte
		AS (SELECT layer_num, node_num, activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1

			UNION ALL

			-- Append 1-valued inputs to match with bias
			SELECT layer_num,
					MAX(node_num) + 1,   -- extra node in next layer
					1    -- activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1
			GROUP BY layer_num),
		layer_cte AS (	
			SELECT lv_layer_iter.layer_num,
					nn.weight_num, 
					-- z = a * W
					SUM(ls.activity * nn.weight) AS result
			FROM prev_layer_cte ls
			JOIN vw_neural_network nn ON nn.node_num = ls.node_num
			WHERE nn.network_id = v_network_id 
				AND nn.layer_num = lv_layer_iter.layer_num   -- correct network layer
			GROUP BY nn.weight_num)
		-- insert result of matrix multiplication in table for this layer
		INSERT INTO tmp_layer_state  (layer_num, node_num, result, activity)
		SELECT layer_num,
				weight_num,
				result,
				CASE WHEN lv_hidden_activation_fn = 'LINEAR'
						THEN result
					 WHEN lv_hidden_activation_fn = 'TANH'
						THEN fn_tanh(result)
					 WHEN lv_hidden_activation_fn = 'RELU'
						THEN fn_relu(result)
					 WHEN lv_hidden_activation_fn = 'SIGMOID'
						THEN fn_sigmoid(result)
					 ELSE result
				END   -- determine activation function for hidden layer(s)
		FROM layer_cte;
			
	END LOOP;	-- end hidden layer loop
	
	
	-- Calculate last layer and network outputs
	WITH
	prev_layer_cte
	AS (SELECT layer_num, node_num, activity
		FROM tmp_layer_state ls
		WHERE ls.layer_num = lv_num_layers - 1

		UNION ALL
		-- Append 1-valued inputs to match with bias
		SELECT layer_num,
				MAX(node_num) + 1,   -- extra node in next layer
				1    -- activity
		FROM tmp_layer_state ls
		WHERE ls.layer_num = lv_num_layers - 1
		GROUP BY layer_num),
	layer_cte 
	AS (SELECT lv_num_layers AS layer_num,
				nn.weight_num,
				-- z = a * W
				SUM(ls.activity * nn.weight) AS result
		FROM prev_layer_cte ls
		JOIN vw_neural_network nn ON nn.node_num = ls.node_num
		WHERE nn.network_id = v_network_id 
			AND nn.layer_num = lv_num_layers   -- correct network layer
		GROUP BY nn.weight_num)
	-- insert result of matrix multiplication in table for this layer
	INSERT INTO tmp_layer_state (layer_num, node_num, result, activity)
	SELECT layer_num,
			weight_num,
			result,
			CASE WHEN lv_output_activation_fn = 'LINEAR'
					THEN result
				 WHEN lv_output_activation_fn = 'TANH'
					THEN fn_tanh(result)
				 WHEN lv_output_activation_fn = 'RELU'
					THEN fn_relu(result)
				 WHEN lv_output_activation_fn = 'SIGMOID'
					THEN fn_sigmoid(result)
				 ELSE result
			END   -- determine activation function for output layer(s)
	FROM layer_cte;
	
	
	IF lv_loss_fn = 'MSE'
	THEN
		RETURN QUERY
			WITH
			statistics_cte 
			AS (
				SELECT label_num,
						label_min_value,
						label_max_value
				FROM dataset ds,
					UNNEST(label_min, label_max) WITH ORDINALITY AS l(label_min_value, label_max_value, label_num)
				WHERE ds.id = lv_dataset_id),
			output_cte
			AS (SELECT CAST(sc.label_num AS int) AS _output_num,
						ls.activity * (sc.label_max_value - sc.label_min_value) + sc.label_min_value AS _output,
						ls.activity,
						(v_label - sc.label_min_value) / (sc.label_max_value - sc.label_min_value) AS label_norm  --* (sc.label_max_value - sc.label_min_value) + sc.label_min_value,
				FROM tmp_layer_state ls
				JOIN statistics_cte sc ON ls.node_num = sc.label_num
				WHERE layer_num = lv_num_layers)
			SELECT _output_num,
					_output,
					_output,
					-- Loss = (1/2) (y - _y)^2
					POWER(label_norm - activity, 2) / 2
			FROM output_cte;
			
			/*SELECT CAST(sc.label_num AS int),
					ls.activity * (sc.label_max_value - sc.label_min_value) + sc.label_min_value,
					NULL::float,
					v_label
			FROM tmp_layer_state ls
			JOIN statistics_cte sc ON ls.node_num = sc.label_num
			WHERE layer_num = lv_num_layers;*/
			
	ELSIF lv_loss_fn = 'CROSS_ENTROPY'
	THEN
		RETURN QUERY
			--o_result_num integer, o_result float, o_sm_result float, o_loss float) 

			-- Choose highest valued activity
			WITH 
			softmax_cte
			AS (SELECT node_num,
						activity,
						EXP(activity) / (SUM(EXP(activity)) OVER ()) AS sm_result,
						ROW_NUMBER() OVER (PARTITION BY layer_num ORDER BY activity DESC) AS row_num
				FROM tmp_layer_state ls
				WHERE layer_num = lv_num_layers)
			SELECT node_num,
					activity,
					sm_result,
					-1*LOG(sm_result)
			FROM softmax_cte
			WHERE row_num = 1   -- Get active node_num corresponding to class number
			ORDER BY node_num;
	ELSE
		RETURN QUERY
			SELECT NULL,NULL,NULL,NULL;
	END IF;
	
	-- Drop temp tables
	DROP TABLE tmp_layer_state;	
	
END;
$$;


--
-- TOC entry 346 (class 1255 OID 16567)
-- Name: fx_sample_feature_select(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fx_sample_feature_select(v_table_name text) RETURNS TABLE(sample_id bigint, feature_num bigint, feature_value double precision)
    LANGUAGE plpgsql
    AS $$
BEGIN

	
	IF UPPER(v_table_name) = 'MNIST'
	THEN
	RETURN QUERY
		WITH 
		samples_cte
		AS (SELECT *,
					ROW_NUMBER() OVER (ORDER BY oid) AS sample_id
			FROM mnist),
		features_cte 
		AS (SELECT s.sample_id,
					f.*
			FROM samples_cte s,
				 UNNEST(avals(hstore(samples_cte) - ARRAY['number', 'label']))
					WITH ORDINALITY f(feature_value, feature_num))
		SELECT sample_id,
				feature_num,
				feature_value
		FROM features_cte
		ORDER BY sample_id, feature_num;
	END IF;
	
	IF UPPER(v_table_name) = 'IRIS'
	THEN
	RETURN QUERY
		WITH 
		samples_cte
		AS (SELECT *,
					id AS sample_id --ROW_NUMBER() OVER (ORDER BY oid) AS sample_id
			FROM iris),
		features_cte 
		AS (SELECT s.sample_id,
					f.*
			FROM samples_cte s,
				 UNNEST(hstore(s) -> ARRAY['sepal_length','sepal_width','petal_length','petal_width'])
					WITH ORDINALITY f(feature_value, feature_num))
		SELECT sample_id,
				feature_num,
				feature_value
		FROM features_cte
		ORDER BY sample_id, feature_num;
	END IF;
	
	
	
END;
$$;


--
-- TOC entry 347 (class 1255 OID 16568)
-- Name: fx_sample_label_select(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fx_sample_label_select(v_table_name text) RETURNS TABLE(sample_id bigint, label_num bigint, label_value double precision)
    LANGUAGE plpgsql
    AS $$
BEGIN

	
	IF v_table_name = 'MNIST'
	THEN
	RETURN QUERY
		WITH 
		samples_cte
		AS (SELECT *,
					ROW_NUMBER() OVER (ORDER BY oid) AS sample_id
			FROM mnist),
		labels_cte 
		AS (SELECT s.sample_id,
					l.*
			FROM samples_cte s,
				UNNEST(hstore(s) -> ARRAY['label']) 
					WITH ORDINALITY l(label_value, label_num))
		SELECT sample_id,
				label_num,
				CAST(label_value AS float)
		FROM labels_cte
		ORDER BY sample_id, label_num;
	END IF;
	
	IF UPPER(v_table_name) = 'IRIS'
	THEN
	RETURN QUERY
		WITH 
		samples_cte
		AS (SELECT id AS sample_id, --ROW_NUMBER() OVER (ORDER BY oid) AS sample_id
					sepal_length,
					sepal_width,
					petal_length,
					petal_width,
					species
			FROM iris),
		label_values_cte 
		AS (SELECT species AS label_value,
				ROW_NUMBER() OVER (ORDER BY species) AS row_num
			FROM samples_cte
			GROUP BY species),
		labels_cte 
		AS (SELECT s.sample_id,
					l.*
			FROM samples_cte s,
				UNNEST(hstore(s) -> ARRAY['species']) 
					WITH ORDINALITY l(label_value, label_num))
		SELECT --l.label_value	
				CAST(l.sample_id AS bigint),
				l.label_num,
				CAST(lv.row_num AS float) AS label_value
		FROM labels_cte l
		JOIN label_values_cte lv ON l.label_value = lv.label_value
		ORDER BY l.sample_id, l.label_num;
	END IF;


	
END;
$$;


--
-- TOC entry 354 (class 1255 OID 16569)
-- Name: fx_test_network(integer, boolean, boolean, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.fx_test_network(v_network_id integer, v_test_samples_only boolean, v_show_samples boolean, v_num_samples integer DEFAULT NULL::integer) RETURNS TABLE(type integer, test_sample_id integer, output_num integer, label_value double precision, output_value double precision, sm_output_value double precision, loss double precision, accuracy double precision)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	/*
	v_network_id integer,
	v_features double precision[]
	*/
	lv_dataset_id int;
	lv_input_activation_fn text;
	lv_hidden_activation_fn text;
	lv_output_activation_fn text;
	lv_normalize_features text;
	lv_normalize_labels text;
	lv_loss_fn text;
	lv_num_layers integer;
	lv_layer_iter RECORD;
BEGIN

	---------------------------
	-- Initialization
	---------------------------
	
	-- Get network information
	SELECT 
		dataset_id,
		activation_functions[1],
		activation_functions[2],
		activation_functions[3],
		loss_function
	INTO
		lv_dataset_id,
		lv_input_activation_fn,
		lv_hidden_activation_fn,
		lv_output_activation_fn,
		lv_loss_fn
	FROM neural_network_architecture
	WHERE id = v_network_id;
	
	-- Calculate layers in network
	SELECT MAX(layer_num)
	INTO lv_num_layers
	FROM neural_network
	WHERE network_id = v_network_id;
	
   	-- Get dataset information
	SELECT
		normalize_features,
		normalize_labels
	INTO
		lv_normalize_features,
		lv_normalize_labels
	FROM dataset
	WHERE id = lv_dataset_id;
	
	
	-- Declare temp tables
	CREATE TEMP TABLE tmp_layer_state
	(
		layer_num int,
		sample_id int,
		node_num int,
		result double precision,   -- result of matrix multiplication before activation fn
		activity double precision   -- result after activation function
	);
	
	CREATE TEMP TABLE tmp_sample_feature_norm
	(
		sample_id int,
		feature_num int,
		feature_value float
	);
	
	CREATE TEMP TABLE tmp_sample_label_norm
	(
		sample_id int,
		label_num int,
		label_value float
	);
	 

	---------------------------
	-- Forward Propagation
	---------------------------
	
	-- Insert input node weights into table
	WITH
	features_cte 
	AS (SELECT s.id AS sample_id, 
				f.feature_num,
				f.feature_value,
				DENSE_RANK() OVER (ORDER BY random()) AS sample_num   -- used if a smaller number of samples is selected
		 FROM sample s,
			UNNEST(s.features) WITH ORDINALITY f(feature_value, feature_num)
		 WHERE s.dataset_id = lv_dataset_id					
			/*--
			AND (v_test_samples_only IS NULL
				 	OR v_test_samples_only = false 
				 	OR (v_test_samples_only = true AND s.training = false))),
			*/
			AND (v_test_samples_only IS NULL OR v_test_samples_only = NOT s.training)),
	statistics_cte
	AS (SELECT f.feature_num,
				 f.feature_mean_value,
				 f.feature_std_value,
				 f.feature_max_value,
				 f.feature_min_value
		FROM dataset d,
			UNNEST(feature_mean, feature_std, feature_max, feature_min)
				WITH ORDINALITY f(feature_mean_value, feature_std_value, feature_max_value, feature_min_value, feature_num)
	   WHERE d.id = lv_dataset_id),
	feature_norm_cte
	AS (SELECT f.sample_id,
				f.feature_num,
				CASE WHEN lv_normalize_features = 'ZSCORE'
						THEN (f.feature_value - COALESCE(s.feature_mean_value,0)) / CASE WHEN (s.feature_std_value = 0 OR s.feature_std_value IS NULL) THEN 1 ELSE s.feature_std_value END   -- normalized value
					WHEN lv_normalize_features = 'MINMAX'
						THEN (f.feature_value - COALESCE(s.feature_max_value,0)) / (s.feature_max_value - s.feature_min_value)   -- normalized value
				END AS feature_value
		FROM features_cte f
		JOIN statistics_cte s ON f.feature_num = s.feature_num
	   	WHERE (sample_num <= v_num_samples OR v_num_samples IS NULL)),   -- choose smaller number of samples
	sample_feature_cte
	AS (SELECT sample_id,
				feature_num,
				feature_value
		 FROM feature_norm_cte

		 UNION ALL

		 -- Augment inputs with 1-valued features to match with bias
		 SELECT sample_id,
				MAX(feature_num) + 1,   -- feature_num
				1   -- feature_value
		 FROM feature_norm_cte
		 GROUP BY sample_id),
	layer_cte
	AS (SELECT 1 AS layer_num,
				sf.sample_id, 
				nn.weight_num, 
				-- z = W * x 
				SUM(nn.weight * sf.feature_value) AS result
		FROM sample_feature_cte sf
		JOIN vw_neural_network nn ON nn.node_num = sf.feature_num   -- line up inputs and weights
		WHERE nn.network_id = v_network_id   -- network id parameter
			AND nn.layer_num = 1   -- input layer
		GROUP BY sf.sample_id, nn.weight_num)
	-- Insert result of matrix multiplication in table for this layer
	INSERT INTO tmp_layer_state (layer_num, sample_id, node_num, result, activity)
	SELECT layer_num,
			sample_id,
			weight_num,
			result,
			CASE WHEN lv_input_activation_fn = 'LINEAR'
					THEN result
				 WHEN lv_input_activation_fn = 'TANH'
					THEN fn_tanh(result)
				 WHEN lv_input_activation_fn = 'RELU'
					THEN fn_relu(result)
				 WHEN lv_input_activation_fn = 'SIGMOID'
					THEN fn_sigmoid(result)
				ELSE result
			END   -- determine activation function for input layer
	FROM layer_cte;

	-- Loop across hidden layers for forward propagation
	FOR lv_layer_iter IN
		SELECT generate_series(2, lv_num_layers) AS layer_num
	LOOP
		WITH
		prev_layer_cte
		AS (SELECT layer_num, sample_id, node_num, activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1

			UNION ALL

			-- Append 1-valued inputs to match with bias
			SELECT layer_num,
					sample_id, 
					MAX(node_num) + 1,   -- extra node in next layer
					1    -- activity
			FROM tmp_layer_state ls
			WHERE ls.layer_num = lv_layer_iter.layer_num - 1
			GROUP BY layer_num, sample_id),
		layer_cte 
		AS (SELECT lv_layer_iter.layer_num,
					ls.sample_id, 
					nn.weight_num,
					-- z = a * W
					SUM(ls.activity * nn.weight) AS result
			FROM prev_layer_cte ls
			JOIN vw_neural_network nn ON nn.node_num = ls.node_num
			WHERE nn.network_id = v_network_id 
				AND nn.layer_num = lv_layer_iter.layer_num   -- correct network layer
			GROUP BY ls.sample_id, nn.weight_num)
		-- insert result of matrix multiplication in table for this layer
		INSERT INTO tmp_layer_state (layer_num, sample_id, node_num, result, activity)
		SELECT layer_num,
				sample_id,
				weight_num,
				result,
				CASE WHEN lv_layer_iter.layer_num = lv_num_layers
				THEN   -- Output layer activations
					CASE WHEN lv_output_activation_fn = 'LINEAR'
							THEN result
						 WHEN lv_output_activation_fn = 'TANH'
							THEN fn_tanh(result)
						 WHEN lv_output_activation_fn = 'RELU'
							THEN fn_relu(result)
						 WHEN lv_output_activation_fn = 'SIGMOID'
							THEN fn_sigmoid(result)
						 ELSE result
					END
				ELSE   -- Hidden layer activations
					CASE WHEN lv_hidden_activation_fn = 'LINEAR'
							THEN result
						 WHEN lv_hidden_activation_fn = 'TANH'
							THEN fn_tanh(result)
						 WHEN lv_hidden_activation_fn = 'RELU'
							THEN fn_relu(result)
						 WHEN lv_hidden_activation_fn = 'SIGMOID'
							THEN fn_sigmoid(result)
						 ELSE result
					END
				END
		FROM layer_cte;

	END LOOP;   -- forward propagation layer loop
	
	
	IF lv_loss_fn = 'MSE'
	THEN
		RETURN QUERY
			WITH
			statistics_cte 
			AS (
				SELECT label_num,
						 label_mean_value,
						 label_std_value,
						 label_min_value,
						 label_max_value
				FROM dataset ds,
					UNNEST(label_mean, label_std, label_max, label_min)
						WITH ORDINALITY l(label_mean_value, label_std_value, label_max_value, label_min_value, label_num)
				WHERE ds.id = lv_dataset_id),
			output_cte
			AS (SELECT s.id AS _sample_id,
						CAST(sc.label_num AS int) AS _output_num,
						CASE WHEN lv_normalize_labels = 'ZSCORE'
								THEN (ls.activity * sc.label_std_value) + sc.label_mean_value
							WHEN lv_normalize_labels = 'MINMAX'
								THEN ls.activity * (sc.label_max_value - sc.label_min_value) + sc.label_min_value 
							ELSE 
								ls.activity 
						END AS _output,   --denormalized prediction
						ls.activity,
						s.label_value AS _label_value,   -- denormalized label
						CASE WHEN lv_normalize_labels = 'ZSCORE'
								THEN (s.label_value - sc.label_mean_value) / sc.label_std_value
							WHEN lv_normalize_labels = 'MINMAX'
								THEN (s.label_value - sc.label_min_value) / (sc.label_max_value - sc.label_min_value)
							ELSE 
								ls.activity 
						END AS _label_value_norm   -- normalized label
				FROM tmp_layer_state ls
				JOIN statistics_cte sc ON ls.node_num = sc.label_num
				JOIN (SELECT s.id, l.label_num, l.label_value
					 	FROM sample s, UNNEST(labels) WITH ORDINALITY l(label_value, label_num)) s
					ON s.id = ls.sample_id AND s.label_num = ls.node_num
				WHERE layer_num = lv_num_layers)

			-- Individual Samples
			SELECT 0 AS _type,
					_sample_id,
					_output_num,
					_label_value,
					_output,
					_output,
					-- Loss = (1/2) (y - _y)^2
					POWER(_label_value_norm - activity, 2) / 2, --POWER(_label_value - activity, 2) / 2,
					100.0 - ABS(_label_value-_output) / _label_value * 100
			FROM output_cte
			WHERE v_show_samples = true
			
			-- Statistics
			UNION ALL
			
			SELECT 1,
					NULL,   --sample_id,
					NULL,   --output_num,
					NULL,   --label_value,
					NULL,   --output,
					NULL,   --output,
					-- Loss = (1/2) (y - _y)^2
					SUM(POWER(_label_value_norm - activity, 2) / 2) / COUNT(*)::float,
					--SUM(ABS(100 - ((_label_value-_output) / _label_value * 100))) / COUNT(*)::float
					100.0 - SUM(ABS(_label_value-_output) / _label_value) * 100 / COUNT(*)::float
			FROM output_cte	
			ORDER BY _type DESC, _label_value, _sample_id, _output_num;
			
	ELSIF lv_loss_fn = 'CROSS_ENTROPY'
	THEN
		RETURN QUERY
			WITH
			softmax_top_cte
			AS (SELECT ls.sample_id, 
						ls.node_num,
						ls.activity,
						-- Z = SUM(EXP(z - mu))  : normalization
						--    mu = MAX(z)
						EXP(ls.activity
							- MAX(ls.activity) OVER (PARTITION BY ls.sample_id)) 
							AS exp_result
				FROM tmp_layer_state ls
				WHERE ls.layer_num = lv_num_layers),
			softmax_cte
			AS (SELECT smt.sample_id, 
						smt.node_num,
						smt.activity,
						-- Si = Zi / SUM(Z)
						smt.exp_result 
							/ SUM(smt.exp_result) OVER (PARTITION BY smt.sample_id)
							AS sm_result,
						ROW_NUMBER() OVER (PARTITION BY smt.sample_id ORDER BY activity DESC) AS row_num
				FROM softmax_top_cte smt)

			-- Individual Samples
			SELECT 0 AS _type,
					sm.sample_id,
					sm.node_num,
					CAST(s.label_value AS INT)::float,
					sm.activity,
					sm.sm_result,
					-1*LN(
						(SELECT sm1.sm_result 
						 FROM softmax_cte sm1 
						 WHERE sm1.sample_id = sm.sample_id 
						 	AND sm1.node_num = CAST(s.label_value AS INT))
					), -- Get correct sample label and match with node_num
					(CASE WHEN CAST(s.label_value AS INT) = sm.node_num THEN 1 ELSE 0 END)::float
			FROM softmax_cte sm,
				LATERAL (SELECT l.label_num, l.label_value
							FROM sample s, 
								UNNEST(labels) WITH ORDINALITY l(label_value, label_num)
							WHERE s.id = sm.sample_id
						) s
			WHERE row_num = 1   -- Get node_num corresponding to class number of highest output
				AND v_show_samples = true
			
			-- Statistics
			UNION ALL
			
			SELECT 1,
					NULL,   -- sample_id
					NULL,   --node_num
					NULL,   --label_value
					NULL,   --activity
					NULL,   --sm_result
					SUM(-1*LN(
						(SELECT sm1.sm_result 
						 FROM softmax_cte sm1 
						 WHERE sm1.sample_id = sm.sample_id 
						 	AND sm1.node_num = CAST(s.label_value AS INT))
					)) / COUNT(*),
					SUM(CASE WHEN CAST(s.label_value AS INT) = sm.node_num THEN 1 ELSE 0 END) / COUNT(*)::float
			FROM softmax_cte sm,
				LATERAL (SELECT l.label_num, l.label_value
							FROM sample s, 
								UNNEST(labels) WITH ORDINALITY l(label_value, label_num)
							WHERE s.id = sm.sample_id
						) s
			WHERE row_num = 1   -- Get node_num corresponding to class number of highest output		
			ORDER BY _type DESC, label_value, sample_id, node_num;
			
	ELSE
		RETURN QUERY
			SELECT NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL;
	END IF;
	
	-- Drop temp tables
	DROP TABLE tmp_layer_state;	
	DROP TABLE tmp_sample_feature_norm;
	DROP TABLE tmp_sample_label_norm;
	
END;
$$;


--
-- TOC entry 348 (class 1255 OID 16571)
-- Name: sp_calculate_dataset_stats(integer); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_calculate_dataset_stats(v_dataset_id integer)
    LANGUAGE plpgsql
    AS $$
--DECLARE 
--	x integer;
BEGIN

	-- Calculate mean and standard deviation for every
	--  feature in the samples.
	-- Then, update dataset with the values
	
	---------------------------------
	-- Calculate Feature Statistics
	---------------------------------
	WITH
	features_cte 
	AS (SELECT s.id AS sample_id, 
				f.feature_num,
				f.feature_value
		 FROM sample s,
			UNNEST(features) WITH ORDINALITY f(feature_value, feature_num)
		 WHERE dataset_id = v_dataset_id),
	 mean_cte
	 AS (SELECT feature_num,
				-- mean = (1/n) * SUM(x)
				SUM(feature_value) / COUNT(feature_num) AS mean
		 FROM features_cte
		 GROUP BY feature_num),
	 std_cte
	 AS (SELECT f.feature_num,
				-- std = SQRT[(1/n) * SUM( (x - mean)^2 )]
				SQRT( SUM(POWER(f.feature_value - m.mean, 2)) / COUNT(f.feature_num)) AS std
		 FROM features_cte f
		 JOIN mean_cte m ON f.feature_num = m.feature_num
		 GROUP BY f.feature_num),
	 range_cte
	 AS (SELECT feature_num,
				 MIN(feature_value) AS min_feature_value,
				 MAX(feature_value) AS max_feature_value
		 FROM features_cte
		 GROUP BY feature_num)
		 
	 UPDATE dataset
	 SET feature_mean = ARRAY(SELECT mean FROM mean_cte ORDER BY feature_num),
		 feature_std = ARRAY(SELECT std FROM std_cte ORDER BY feature_num),
		 feature_max = ARRAY(SELECT max_feature_value FROM range_cte ORDER BY feature_num),
		 feature_min = ARRAY(SELECT min_feature_value FROM range_cte ORDER BY feature_num)
	 WHERE id = v_dataset_id;

	---------------------------------
	-- Calculate Label Statistics
	---------------------------------
	WITH
	labels_cte 
	AS (SELECT s.id AS sample_id, 
				l.label_num,
				l.label_value
		 FROM sample s,
			UNNEST(labels) WITH ORDINALITY l(label_value, label_num)
		 WHERE dataset_id = v_dataset_id),
	 mean_cte
	 AS (SELECT label_num,
				-- mean = (1/n) * SUM(x)
				SUM(label_value) / COUNT(label_num) AS mean
		 FROM labels_cte
		 GROUP BY label_num),
	 std_cte
	 AS (SELECT l.label_num,
				-- std = SQRT[(1/n) * SUM( (x - mean)^2 )]
				SQRT( SUM(POWER(l.label_value - m.mean, 2)) / COUNT(l.label_num)) AS std
		 FROM labels_cte l
		 JOIN mean_cte m ON l.label_num = m.label_num
		 GROUP BY l.label_num),
	 range_cte
	 AS (SELECT label_num,
				 MIN(label_value) AS min_label_value,
				 MAX(label_value) AS max_label_value
		 FROM labels_cte
		 GROUP BY label_num)
		 
	 UPDATE dataset
	 SET label_mean = ARRAY(SELECT mean FROM mean_cte ORDER BY label_num),
		 label_std = ARRAY(SELECT std FROM std_cte ORDER BY label_num),
		 label_max = ARRAY(SELECT max_label_value FROM range_cte ORDER BY label_num),
		 label_min = ARRAY(SELECT min_label_value FROM range_cte ORDER BY label_num)
	 WHERE id = v_dataset_id;
		 
END;
$$;


--
-- TOC entry 349 (class 1255 OID 16572)
-- Name: sp_delete_dataset(integer); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_delete_dataset(v_dataset_id integer)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	lv_network_id int;
BEGIN

	DELETE FROM neural_network nn
	WHERE nn.network_id IN (SELECT nna.id 
							FROM neural_network_architecture nna
							WHERE nna.dataset_id = v_dataset_id);
	
	DELETE FROM neural_network_architecture
	WHERE dataset_id = v_dataset_id;
	
	DELETE FROM sample
	WHERE dataset_id = v_dataset_id;
	
	DELETE FROM dataset
	WHERE id = v_dataset_id;

END;
$$;


--
-- TOC entry 350 (class 1255 OID 16573)
-- Name: sp_delete_network(integer); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_delete_network(v_network_id integer)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	lv_network_id int;
BEGIN

	DELETE FROM neural_network WHERE network_id = v_network_id;
	
	DELETE FROM neural_network_architecture WHERE id = v_network_id;

END;
$$;


--
-- TOC entry 357 (class 1255 OID 16574)
-- Name: sp_insert_dataset(text, text, text, text, text, text); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_insert_dataset(v_dataset_name text, v_source_table_name text, v_feature_column_names text, v_label_column_names text, v_normalize_features text, v_normalize_labels text)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	lv_dataset_id int;
BEGIN

	INSERT INTO dataset (name, normalize_features, normalize_labels)
	SELECT v_dataset_name,
			CASE WHEN v_normalize_features = '' THEN NULL ELSE v_normalize_features END,
			CASE WHEN v_normalize_labels = '' THEN NULL ELSE v_normalize_labels END
	RETURNING id
	INTO lv_dataset_id;

	EXECUTE FORMAT('
		WITH 
		samples_cte
		AS (SELECT *,
				   ROW_NUMBER() OVER () AS sample_num
			FROM %s),
		features_cte 
		AS (SELECT s.sample_num,
					f.feature_num,
					f.feature_value,
					DENSE_RANK() OVER (ORDER BY f.feature_value) AS rank_num
			FROM samples_cte s,
				UNNEST(hstore(s) -> ARRAY[%s]) 
					WITH ORDINALITY f(feature_value, feature_num)),
		labels_cte 
		AS (SELECT sample_num,
					l.label_num,
					l.label_value,
					DENSE_RANK() OVER (ORDER BY l.label_value) AS rank_num
			FROM samples_cte s,
				UNNEST(hstore(s) -> ARRAY[%s]) 
					WITH ORDINALITY l(label_value, label_num))
		INSERT INTO sample
		(
			dataset_id,
			features,
			labels,
			training
		)
		SELECT %s,
				ARRAY(SELECT CASE WHEN fn_isnumeric(feature_value) 
									THEN feature_value::float
								ELSE rank_num::float
							END 
						FROM features_cte f
						WHERE f.sample_num = s.sample_num
						ORDER BY feature_num),
				ARRAY(SELECT CASE WHEN fn_isnumeric(label_value) 
									THEN label_value::float
								ELSE rank_num::float
							END 
						FROM labels_cte l
						WHERE l.sample_num = s.sample_num
						ORDER BY label_num),
				true
	   FROM samples_cte s
		', quote_ident(v_source_table_name), 
			CONCAT('''', REPLACE(v_feature_column_names, ',', ''','''), ''''),
			CONCAT('''', REPLACE(v_label_column_names, ',', ''','''), ''''),
			lv_dataset_id);
			
	CALL sp_calculate_dataset_stats(lv_dataset_id);

END;
$$;


--
-- TOC entry 353 (class 1255 OID 16575)
-- Name: sp_insert_network(text, integer, text[], text, integer, integer, integer, integer); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_insert_network(name text, v_dataset_id integer, v_activation_functions text[], v_loss_function text, v_num_inputs integer, v_num_hidden_layers integer, v_num_hidden_nodes integer, v_num_outputs integer)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	lv_network_id int;
BEGIN

	-- Create network record
	INSERT INTO neural_network_architecture(name,
											dataset_id,
										   	activation_functions,
										    loss_function,
										    num_hidden_layers,
										    layer_sizes)
   	SELECT name,
			v_dataset_id,
			v_activation_functions,
		   	v_loss_function,
			v_num_hidden_layers,
			ARRAY[v_num_inputs, v_num_hidden_nodes, v_num_outputs]
	RETURNING id
	INTO lv_network_id;
	
	
	-- Generate input weights. 1 x (num_inputs+1) x num_hidden_nodes
	INSERT INTO neural_network(network_id,
								layer_num,
								node_num,
								weight_num,
								weight)
	SELECT lv_network_id, 
			1,   -- layer_num = 1
			n,   -- node_num
			w,   -- weight_num
			RANDOM() * 0.1 - 0.05 AS weight   -- init with random weight
	FROM generate_series(1,v_num_inputs) n,
		generate_series(1,v_num_hidden_nodes) w
	UNION ALL
	SELECT lv_network_id, 
			1,   -- layer_num = 1
			v_num_inputs + 1,   -- node_num
			w,   -- weight_num
			0.0   -- init bias with a small value
	FROM generate_series(1,v_num_hidden_nodes) w
	ORDER BY n,w;
	

	-- Generate hidden node weights. v_num_hidden_layers x (num_hidden_nodes+1) x num_hidden_nodes
	INSERT INTO neural_network(network_id,
								layer_num,
								node_num,
							    weight_num,
								weight)
	SELECT lv_network_id, 
			l,   -- layer_num
			n,   -- node_num
			w,   -- weight_num
			RANDOM() * 0.1 - 0.05 as weight   -- init weight with random value
	FROM generate_series(2, v_num_hidden_layers) l,
		generate_series(1, v_num_hidden_nodes) n,
		generate_series(1, v_num_hidden_nodes) w
	UNION ALL
	SELECT lv_network_id, 
			l,   -- layer_num
			v_num_hidden_nodes + 1,   -- node_num
			w,   -- weight_num
			0.0   -- init bias with small value
	FROM generate_series(2, v_num_hidden_layers) l,
			generate_series(1, v_num_hidden_nodes) w
	ORDER BY l, n, w;
	

	-- Generate output node weights. 1 x num_hidden_nodes x num_outputs
	INSERT INTO neural_network(network_id,
								layer_num,
								node_num,
							    weight_num,
								weight)
	SELECT lv_network_id, 
			v_num_hidden_layers + 1,   -- layer_num
			n,   -- node_num
			w,   -- weight_num
			RANDOM() * 0.1 - 0.05   -- init weight with random value
	FROM generate_series(1, v_num_hidden_nodes) n,
		generate_series(1, v_num_outputs) w
	UNION ALL
	SELECT lv_network_id, 
			v_num_hidden_layers + 1,   -- layer_num
			v_num_hidden_nodes + 1,   -- node_num
			w,   -- weight_num
			0.0   -- init bias with small value
	FROM generate_series(1, v_num_outputs) w
	ORDER BY n, w;

END;
$$;


--
-- TOC entry 351 (class 1255 OID 16576)
-- Name: sp_normalize_dataset(integer); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_normalize_dataset(v_dataset_id integer)
    LANGUAGE plpgsql
    AS $$
--DECLARE 
--	x integer;
BEGIN

	-- Calculate mean and standard deviation for every
	--  feature in the samples.
	-- Then, update dataset with the values
	
	---------------------------------
	-- Calculate Feature Statistics
	---------------------------------
	WITH
	features_cte 
	AS (SELECT s.id AS sample_id, 
				f.feature_num,
				f.feature_value
		 FROM sample s,
			UNNEST(features) WITH ORDINALITY f(feature_value, feature_num)
		 WHERE dataset_id = v_dataset_id),
	 mean_cte
	 AS (SELECT feature_num,
				-- mean = (1/n) * SUM(x)
				SUM(feature_value) / COUNT(feature_num) AS mean
		 FROM features_cte
		 GROUP BY feature_num),
	 std_cte
	 AS (SELECT f.feature_num,
				-- std = SQRT[(1/n) * SUM( (x - mean)^2 )]
				SQRT( SUM(POWER(f.feature_value - m.mean, 2)) / COUNT(f.feature_num)) AS std
		 FROM features_cte f
		 JOIN mean_cte m ON f.feature_num = m.feature_num
		 GROUP BY f.feature_num)
	 UPDATE dataset
	 SET feature_mean = ARRAY(SELECT mean FROM mean_cte ORDER BY feature_num),
		 feature_std = ARRAY(SELECT std FROM std_cte ORDER BY feature_num)
	 WHERE id = v_dataset_id;
	 
	---------------------------------
	-- Update Feature Norm
	---------------------------------
	WITH
	cte 
	AS (SELECT s.id,
			f.feature_value,
			(f.feature_value - d.mean) / d.std AS feature_norm,
			f.feature_num
		FROM sample s,
			UNNEST(features) WITH ORDINALITY f(feature_value,feature_num),
			(SELECT s.* 
			  FROM dataset d, UNNEST(feature_mean,feature_std) WITH ORDINALITY s(mean,std,feature_num)
			  WHERE d.id = 2) d
		WHERE s.dataset_id = 2
	   		AND f.feature_num = d.feature_num)
	UPDATE sample s
	SET features_norm = ARRAY(SELECT feature_norm FROM cte c WHERE c.id = s.id)
		--,features_norm = ARRAY((f.feature_value - d.mean) / d.std) 
;

	---------------------------------
	-- Calculate Label Statistics
	---------------------------------
	WITH
	label_cte
	AS (SELECT s.id AS sample_id, 
				l.label_num,
				l.label_value
		 FROM sample s,
			UNNEST(labels) WITH ORDINALITY l(label_value, label_num)
		 WHERE s.dataset_id = v_dataset_id),
	 range_cte
	 AS (SELECT label_num,
				MIN(label_value) AS min_label_value,
				MAX(label_value) AS max_label_value
		FROM label_cte
		GROUP BY label_num)
	 UPDATE dataset
	 SET label_min = ARRAY(SELECT min_label_value FROM range_cte ORDER BY label_num),
		 label_max = ARRAY(SELECT max_label_value FROM range_cte ORDER BY label_num)
	 WHERE id = v_dataset_id;
		 
END;
$$;


--
-- TOC entry 356 (class 1255 OID 16577)
-- Name: sp_reset_network(integer, boolean); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_reset_network(v_network_id integer, v_setseed boolean DEFAULT false)
    LANGUAGE plpgsql
    AS $$
BEGIN
	IF v_setseed
	THEN
		SET LOCAL seed = 0.5;   -- set random seed for current transaction
	END IF;
	
	WITH
	num_weights_cte
	AS (SELECT layer_num, 
				MAX(node_num) as max_node_num
		FROM neural_network
		WHERE network_id = v_network_id
		GROUP BY layer_num)
		
	UPDATE neural_network nn
	SET weight = CASE WHEN nn.node_num = mw.max_node_num
						THEN 0.0 --0.1
					ELSE RANDOM() * 0.1 - 0.05
				END
	FROM num_weights_cte mw
	WHERE nn.network_id = v_network_id
		AND nn.layer_num = mw.layer_num;
		
	/*SELECT nn.layer_num,
			nn.node_num,
			nn.weight_num,
			nn.weight,
			CASE when nn.node_num = mw.max_node_num
					THEN 'bias'
				ELSE 'weight'
			END as type
	FROM neural_network nn
	JOIN num_weights_cte mw on nn.layer_num = mw.layer_num
	WHERE network_id=v_network_id*/
	
END;
$$;


--
-- TOC entry 352 (class 1255 OID 16578)
-- Name: sp_set_training_samples(integer, double precision); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_set_training_samples(v_network_id integer, v_percent_test double precision)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	lv_percent_test float;   -- Value [0:1]
BEGIN

	-- Percent validation
	IF v_percent_test < 0 OR v_percent_test > 100
	THEN
		RAISE NOTICE 'Percent (%) must be between 0 and 100. Defaulting to %', v_percent_test, 15;
		lv_percent_test := 0.15;
	ELSE
		lv_percent_test := v_percent_test / 100.0;
	END IF;
	
	WITH
	dataset_cte
	AS (SELECT nna.dataset_id
	  	FROM neural_network_architecture nna
	   	WHERE nna.id = v_network_id),
	sample_cte
	AS (SELECT s.id AS sample_id,
				ROW_NUMBER() OVER (ORDER BY random()) AS sample_row
	   	FROM sample s
	   	JOIN dataset_cte ds ON s.dataset_id = ds.dataset_id)
		
	UPDATE sample s
	SET training = CASE WHEN sample_row <= (SELECT COUNT(*) * lv_percent_test FROM sample_cte)
							THEN false
						ELSE true
					END
	FROM sample_cte sc
	WHERE s.id = sc.sample_id;


END;
$$;


--
-- TOC entry 355 (class 1255 OID 16579)
-- Name: sp_tmp(text); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_tmp(v_param text)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	lv_time float := 1.5;
BEGIN

	RAISE NOTICE 'MY NOTICE 1 (%)', v_param;
	PERFORM pg_sleep(lv_time);
	RAISE NOTICE 'MY NOTICE 2';
	PERFORM pg_sleep(lv_time);
	RAISE NOTICE 'MY NOTICE 3';
	PERFORM pg_sleep(lv_time);
	RAISE NOTICE 'MY NOTICE 4';
	PERFORM pg_sleep(lv_time);
	RAISE NOTICE 'MY NOTICE 5';
	PERFORM pg_sleep(lv_time);
	RAISE NOTICE 'MY NOTICE 6';
	PERFORM pg_sleep(lv_time);
	RAISE NOTICE 'MY NOTICE 7';
	PERFORM pg_sleep(lv_time);
	RAISE NOTICE 'MY NOTICE 8 (%)', v_param;
END;
$$;


--
-- TOC entry 358 (class 1255 OID 16580)
-- Name: sp_train_network(integer, integer, integer, double precision); Type: PROCEDURE; Schema: public; Owner: -
--

CREATE PROCEDURE public.sp_train_network(v_network_id integer, v_num_epochs integer DEFAULT 1, v_batch_size integer DEFAULT 32, v_learning_rate double precision DEFAULT 0.1)
    LANGUAGE plpgsql
    AS $$
DECLARE 
	/*
	v_network_id integer := 16;
	lv_dataset_id integer := 6;
	v_num_epochs integer := 1;
	v_batch_size float := 24.0;
	v_learning_rate double precision := 0.1;
	*/
	lv_start_time time;
	lv_dataset_id integer;
	lv_num_layers integer;
	lv_num_samples int;
	lv_num_batches int;
	lv_input_activation_fn text;
	lv_hidden_activation_fn text;
	lv_output_activation_fn text;
	lv_normalize_features text;
	lv_normalize_labels text;
	lv_loss_fn text;
	lv_layer_iter RECORD;
	lv_last_epoch_start_time time;
	lv_epoch_loss float := 0.0;
	iter RECORD;
BEGIN

	---------------------------
	-- Initialization
	---------------------------
	
	-- Get network information
	SELECT 
		dataset_id,
		activation_functions[1],
		activation_functions[2],
		activation_functions[3],
		loss_function
	INTO
		lv_dataset_id,
		lv_input_activation_fn,
		lv_hidden_activation_fn,
		lv_output_activation_fn,
		lv_loss_fn
	FROM neural_network_architecture
	WHERE id = v_network_id;
	
	-- Get dataset information
	SELECT
		normalize_features,
		normalize_labels
	INTO
		lv_normalize_features,
		lv_normalize_labels
	FROM dataset
	WHERE id = lv_dataset_id;
	
	-- Calculate layers in network
	SELECT MAX(layer_num)
	INTO lv_num_layers
	FROM neural_network
	WHERE network_id = v_network_id;
	
	-- Calculate number of batches
	SELECT COUNT(*) / v_batch_size
	INTO lv_num_batches
	FROM sample
	WHERE dataset_id = lv_dataset_id;
	
	RAISE NOTICE 'Network:%, Dataset:%	 Layers:%, [%:%:%] Loss:%, Batches:%, NF:%, NL:%',v_network_id, lv_dataset_id, lv_num_layers, 
			lv_input_activation_fn, lv_hidden_activation_fn, lv_output_activation_fn, lv_loss_fn, lv_num_batches, lv_normalize_features, lv_normalize_labels;
	
	-- Declare temp tables
	CREATE TEMP TABLE tmp_sample_feature
	(
		sample_id int,
		batch_num int DEFAULT 1,
		feature_num int,
		feature_value float
	);
	
	CREATE TEMP TABLE tmp_sample_label
	(
		sample_id int,
		label_num int,
		label_value float
	);
	
	CREATE TEMP TABLE tmp_layer_state
	(
		layer_num int,
		sample_id int,
		node_num int,
		result float,   -- result of matrix multiplication before activation fn
		activity float   -- result after activation function
	);

	CREATE TEMP TABLE tmp_delta
	(
		layer_num int,
		sample_id int,
		node_num int,
		delta float
	);
	
	SELECT clock_timestamp()::time INTO lv_start_time;
	RAISE NOTICE 'Begin (%)', lv_start_time;

	---------------------------
	-- Features
	---------------------------
	IF lv_normalize_features IS NOT NULL
	THEN
		-- Normalize features with v = (x - mean) / std
		WITH
		features_cte 
		AS (SELECT s.id AS sample_id, 
					f.feature_num,
					f.feature_value
			 FROM sample s,
				UNNEST(features) WITH ORDINALITY f(feature_value, feature_num)
			 WHERE s.dataset_id = lv_dataset_id
			 	AND s.training = true),
		rand_cte
		AS (SELECT sample_id,
					random() AS rand
			FROM features_cte
		   GROUP BY sample_id),
		statistics_cte
		AS (SELECT f.feature_num,
					 f.feature_mean_value,
					 f.feature_std_value,
					 f.feature_max_value,
					 f.feature_min_value
			FROM dataset d,
				UNNEST(feature_mean, feature_std, feature_max, feature_min) 
					WITH ORDINALITY f(feature_mean_value, feature_std_value, feature_max_value, feature_min_value, feature_num)
		   WHERE d.id = lv_dataset_id)
		INSERT INTO tmp_sample_feature (sample_id,
									batch_num,
									feature_num,
									feature_value)
		SELECT f.sample_id,
				NTILE(lv_num_batches) OVER (ORDER BY r.rand),
				f.feature_num,
				CASE WHEN lv_normalize_features = 'ZSCORE'
						-- EQUATION (15)
						THEN (f.feature_value - COALESCE(s.feature_mean_value,0)) / CASE WHEN (s.feature_std_value = 0 OR s.feature_std_value IS NULL) THEN 1 ELSE s.feature_std_value END   -- normalized value
					WHEN lv_normalize_features = 'MINMAX'
						-- EQUATION (16)
						THEN (f.feature_value - COALESCE(s.feature_min_value,0)) / (s.feature_max_value - s.feature_min_value)   -- normalized value
				END
		FROM features_cte f
		JOIN statistics_cte s ON f.feature_num = s.feature_num
		JOIN rand_cte r ON f.sample_id = r.sample_id;		
	ELSE
		-- Do not normalize features
		WITH
		features_cte 
		AS (SELECT s.id AS sample_id, 
					f.feature_num,
					f.feature_value
			 FROM sample s,
				UNNEST(features) WITH ORDINALITY f(feature_value, feature_num)
			 WHERE s.dataset_id = lv_dataset_id
			 	AND s.training = true),
		rand_cte
		AS (SELECT sample_id,
					random() AS rand
			FROM features_cte
		   GROUP BY sample_id)
		INSERT INTO tmp_sample_feature (sample_id,
									batch_num,
									feature_num,
									feature_value)
		SELECT f.sample_id,
				NTILE(lv_num_batches) OVER (ORDER BY r.rand),
				f.feature_num,
				f.feature_value
		FROM features_cte f
		JOIN rand_cte r ON f.sample_id = r.sample_id;
	END IF;
	 

	---------------------------
	-- Labels
	---------------------------
	IF lv_normalize_labels IS NOT NULL
	THEN		 
		-- Normalize labels with v = (x - min) / (max - min)
		WITH
		label_cte
		AS (SELECT s.id AS sample_id, 
					l.label_num,
					l.label_value
			 FROM sample s,
				UNNEST(labels) WITH ORDINALITY l(label_value, label_num)
			 WHERE s.dataset_id = lv_dataset_id
			 	AND s.training = true),
		statistics_cte
		AS (SELECT label_num,
					 label_mean_value,
					 label_std_value,
					 label_min_value,
					 label_max_value
			FROM dataset d,
				UNNEST(label_mean, label_std, label_min, label_max) 
					WITH ORDINALITY l(label_mean_value, label_std_value, label_min_value, label_max_value, label_num)
		   WHERE d.id = lv_dataset_id)
		INSERT INTO tmp_sample_label (sample_id,
									label_num,
									label_value)
		SELECT l.sample_id,
				l.label_num,
				CASE WHEN lv_normalize_labels = 'ZSCORE'
						-- EQUATION (15)
						THEN (l.label_value - s.label_mean_value) / s.label_std_value   -- normalized value
					WHEN lv_normalize_labels = 'MINMAX'
						-- EQUATION (16)
						THEN (l.label_value - s.label_min_value) / (s.label_max_value - s.label_min_value)   -- normalized value
				END
		FROM label_cte l
		JOIN statistics_cte s ON l.label_num = s.label_num;
	ELSE
		-- Do not normalize labels
		WITH
		label_cte
		AS (SELECT s.id AS sample_id, 
					l.label_num,
					l.label_value
			 FROM sample s,
				UNNEST(labels) WITH ORDINALITY l(label_value, label_num)
			 WHERE s.dataset_id = lv_dataset_id
			 	AND s.training = true)
		INSERT INTO tmp_sample_label (sample_id,
									label_num,
									label_value)
		SELECT l.sample_id,
				l.label_num,
				l.label_value
		FROM label_cte l;
	END IF;

/*for lv_layer_iter in 
	SELECT * from tmp_sample_feature sf
	order by sample_id,feature_num
	limit 100
loop
	raise notice 'S%	F%	%', lv_layer_iter.sample_id, lv_layer_iter.feature_num,lv_layer_iter.feature_value;
end loop;*/

	--Optionally create indices
	IF 1=1
	THEN
		CREATE INDEX IX_tmp_sample_feature 
		ON tmp_sample_feature (batch_num)
		WITH (fillfactor = 100);   --static table once made
		
		CREATE INDEX IX_tmp_sample_label
		ON tmp_sample_label (sample_id, label_num)
		WITH (fillfactor = 100);   --static table once made
		
		--CREATE INDEX IX_tmp_sample_label_sample_id
		--ON tmp_sample_label (sample_id)
		--WITH (fillfactor = 100);   --static table once made
		
		CREATE INDEX IX_tmp_layer_state
		ON tmp_layer_state (layer_num);
	END IF;
	

	RAISE NOTICE 'Beginning Training (%). Duration (%)', clock_timestamp()::time, clock_timestamp()::time - lv_start_time;

	
	-- Loop for each epoch
	FOR lv_epoch_num IN 1..v_num_epochs LOOP

		RAISE NOTICE 'Epoch % (%), since last (%)', lv_epoch_num, clock_timestamp()::time, (clock_timestamp()::time - lv_last_epoch_start_time);
		
		SELECT clock_timestamp()::time
		INTO lv_last_epoch_start_time;
		
		-- Loop for each batch
		FOR lv_batch_num IN 1..lv_num_batches LOOP
		
			---------------------------
			-- Forward Propagation
			---------------------------

--RAISE NOTICE '	Begin Forward: %', (clock_timestamp()::time - lv_last_epoch_start_time);

			-- Calculate inputs multiplied with first layer weights
			WITH 
			sample_feature_cte
			AS (SELECT sample_id,
					 		batch_num,
					 		feature_num,
					 		feature_value
					 FROM tmp_sample_feature
					 WHERE batch_num = lv_batch_num
					 
					 UNION ALL
						
					 -- Augment inputs with 1-valued features to match with bias
					 SELECT sample_id,
					 		batch_num,
							MAX(feature_num) + 1,   -- feature_num
							1   -- feature_value
					 FROM tmp_sample_feature
					 WHERE batch_num = lv_batch_num
					 GROUP BY sample_id, batch_num),
			layer_cte AS (
				SELECT 1 AS layer_num,
						sf.sample_id, 
						nn.weight_num, 
						-- z = W * x 
						SUM(nn.weight * sf.feature_value) AS result   -- EQUATION (1)
				FROM sample_feature_cte sf
				JOIN neural_network nn ON nn.node_num = sf.feature_num   -- line up inputs and weights
				WHERE nn.network_id = v_network_id   -- network id parameter
					AND nn.layer_num = 1   -- input layer
				GROUP BY sf.sample_id, nn.weight_num)
			-- Insert result of matrix multiplication in table for this layer
			INSERT INTO tmp_layer_state (layer_num, sample_id, node_num, result, activity)
			SELECT layer_num,
					sample_id,
					weight_num,
					result,
					-- EQUATION (2)
					CASE WHEN lv_input_activation_fn = 'LINEAR'
							THEN result  -- EQUATION (17)
						 WHEN lv_input_activation_fn = 'TANH'
							THEN fn_tanh(result)  -- EQUATION (21)
						 WHEN lv_input_activation_fn = 'RELU'
							THEN fn_relu(result)  -- EQUATION (23)
						 WHEN lv_input_activation_fn = 'SIGMOID'
							THEN fn_sigmoid(result)  -- EQUATION (19)
						ELSE result
					END   -- determine activation function for input layer
			FROM layer_cte;
			
--RAISE NOTICE '	End Fwd Inputs: %', (clock_timestamp()::time - lv_last_epoch_start_time);

			-- Loop across hidden layers for forward propagation
			FOR lv_layer_iter IN
				SELECT generate_series(2, lv_num_layers) AS layer_num
			LOOP
				WITH
				prev_layer_cte
				AS (SELECT layer_num, sample_id, node_num, activity
				   	FROM tmp_layer_state ls
				   	WHERE ls.layer_num = lv_layer_iter.layer_num - 1
				   
				   	UNION ALL
					
					-- Append 1-valued inputs to match with bias
					SELECT layer_num,
							sample_id, 
							MAX(node_num) + 1,   -- extra node in next layer
							1    -- activity
					FROM tmp_layer_state ls
				   	WHERE ls.layer_num = lv_layer_iter.layer_num - 1
					GROUP BY layer_num, sample_id),
				layer_cte 
				AS (SELECT lv_layer_iter.layer_num,
							ls.sample_id, 
							nn.weight_num,
							-- z = a * W
							SUM(ls.activity * nn.weight) AS result  -- EQUATION (1)
					FROM prev_layer_cte ls
					JOIN neural_network nn ON nn.node_num = ls.node_num
					WHERE nn.network_id = v_network_id 
						AND nn.layer_num = lv_layer_iter.layer_num   -- correct network layer
					GROUP BY ls.sample_id, nn.weight_num)
				-- insert result of matrix multiplication in table for this layer
				INSERT INTO tmp_layer_state (layer_num, sample_id, node_num, result, activity)
				SELECT layer_num,
						sample_id,
						weight_num,
						result,
						CASE WHEN lv_layer_iter.layer_num = lv_num_layers
					    -- EQUATION (2)
						THEN   -- Output layer activations
							CASE WHEN lv_output_activation_fn = 'LINEAR'
									THEN result  -- EQUATION (17)
								 WHEN lv_output_activation_fn = 'TANH'
									THEN fn_tanh(result)  -- EQUATION (21)
								 WHEN lv_output_activation_fn = 'RELU'
									THEN fn_relu(result)  -- EQUATION (23)
								 WHEN lv_output_activation_fn = 'SIGMOID'
									THEN fn_sigmoid(result)  -- EQUATION (19)
								 ELSE result
							END
						ELSE   -- Hidden layer activations
							CASE WHEN lv_hidden_activation_fn = 'LINEAR'
									THEN result  -- EQUATION (17)
								 WHEN lv_hidden_activation_fn = 'TANH'
									THEN fn_tanh(result)  -- EQUATION (21)
								 WHEN lv_hidden_activation_fn = 'RELU'
									THEN fn_relu(result)  -- EQUATION (23)
								 WHEN lv_hidden_activation_fn = 'SIGMOID'
									THEN fn_sigmoid(result)  -- EQUATION (19)
								 ELSE result
							END
						END
				FROM layer_cte;

			END LOOP;   -- forward propagation layer loop

			-- Calculate last layer and network outputs
			/*WITH
			prev_layer_cte
			AS (SELECT layer_num, sample_id, node_num, activity
				FROM tmp_layer_state ls
				WHERE ls.layer_num = lv_num_layers - 1

				UNION ALL
				-- Append 1-valued inputs to match with bias
				SELECT layer_num,
						sample_id, 
						MAX(node_num) + 1,   -- extra node in next layer
						1    -- activity
				FROM tmp_layer_state ls
				WHERE ls.layer_num = lv_num_layers - 1
				GROUP BY layer_num, sample_id),
			layer_cte 
			AS (SELECT lv_num_layers AS layer_num,
						ls.sample_id, 
						nn.weight_num,
						-- z = a * W
						SUM(ls.activity * nn.weight) AS result
				FROM prev_layer_cte ls
				JOIN neural_network nn ON nn.node_num = ls.node_num
				WHERE nn.network_id = v_network_id 
					AND nn.layer_num = lv_num_layers   -- correct network layer
				GROUP BY ls.sample_id, nn.weight_num)
			-- insert result of matrix multiplication in table for this layer
			INSERT INTO tmp_layer_state (layer_num, sample_id, node_num, result, activity)
			SELECT layer_num,
					sample_id,
					weight_num,
					result,
					CASE WHEN lv_output_activation_fn = 'LINEAR'
							THEN result
						 WHEN lv_output_activation_fn = 'TANH'
							THEN fn_tanh(result)
						 WHEN lv_output_activation_fn = 'RELU'
							THEN fn_relu(result)
						 WHEN lv_output_activation_fn = 'SIGMOID'
							THEN fn_sigmoid(result)
						 ELSE result
					END   -- determine activation function for hidden layer(s)
			FROM layer_cte;*/

			/*raise notice 'layers:';
			for lv_layer_iter in 
				SELECT * from tmp_layer_state sf
				order by layer_num,sample_id, node_num
			loop
				raise notice '	L%	S%	N%	%	%', lv_layer_iter.layer_num, lv_layer_iter.sample_id,lv_layer_iter.node_num, lv_layer_iter.result, lv_layer_iter.activity;
			end loop;*/

--RAISE NOTICE '	End Forward: %', (clock_timestamp()::time - lv_last_epoch_start_time);
		
			---------------------------
			-- Back Propagation
			---------------------------

			-- Count number of samples
			SELECT COUNT(DISTINCT sample_id)
			INTO lv_num_samples
			FROM tmp_sample_feature
			WHERE batch_num = lv_batch_num;

			-- Insert output layer deltas corresponding to loss function
			IF lv_loss_fn = 'MSE'
			THEN
				--RAISE NOTICE '	MSE Loss';

				-- Calculate Mean Squared Error loss
				WITH
				layer_cte
				AS (SELECT layer_num, sample_id, node_num, result, activity
					FROM tmp_layer_state ls
					WHERE ls.layer_num = lv_num_layers

					/*UNION ALL
					
					-- Append 1-valued inputs to match with bias
					SELECT layer_num,
							sample_id, 
							MAX(node_num) + 1,   -- extra node in next layer
							1,    -- result
							1    -- activity
					FROM tmp_layer_state ls
					WHERE ls.layer_num = lv_num_layers
					GROUP BY layer_num, sample_id*/)
				INSERT INTO tmp_delta (layer_num, sample_id, node_num, delta)
				SELECT ls.layer_num,
						ls.sample_id,
						ls.node_num,
						-- delta = -(y - y^) x activation_prime(z)
						-- -1 * (sl.label_value - ls.activity) 
						-- EQUATION (3)
						(ls.activity - sl.label_value) -- flipping values introduces minus sign
							  * CASE WHEN lv_output_activation_fn = 'LINEAR'
										THEN 1--result  -- EQUATION (18)
									 WHEN lv_output_activation_fn = 'TANH'
										THEN fn_tanh_prime(ls.result)  -- EQUATION (22)
									 WHEN lv_output_activation_fn = 'RELU'
										THEN fn_relu_prime(ls.result)  -- EQUATION (24)
									 WHEN lv_output_activation_fn = 'SIGMOID'
										THEN fn_sigmoid_prime(ls.result)  -- EQUATION (20)
									 ELSE 1--result
								END   -- determine activation function for output layer
				FROM layer_cte ls
				JOIN tmp_sample_label sl ON ls.sample_id = sl.sample_id
					AND ls.node_num = sl.label_num;   -- if multiple labels

				--Print MSE loss
				--FOR iter in
					-- Loss = (0.5)(y - y_)^2
					WITH
					loss_cte
					AS (
					SELECT --SQRT(SUM((sl.label_value - ls.activity)*(sl.label_value - ls.activity) / 2.0) / COUNT(*)) AS avg_loss
						-- EQUATION (7)
						SUM(POWER(sl.label_value - ls.activity, 2)) / 2.0 / COUNT(*) AS avg_mse_loss
					FROM tmp_layer_state ls
					JOIN tmp_sample_label sl ON ls.sample_id = sl.sample_id
						AND ls.node_num = sl.label_num   -- if multiple labels
					WHERE ls.layer_num = lv_num_layers
					)
					SELECT lv_epoch_loss + avg_mse_loss   -- accumulate loss across batches to average later
					INTO lv_epoch_loss
					FROM loss_cte;
				--LOOP
				--	RAISE NOTICE '	Epoch % Batch %: Avg MSE Loss = %', lv_epoch_num, lv_batch_num, iter.avg_mse_loss;
				--END LOOP;

			ELSIF lv_loss_fn = 'CROSS_ENTROPY'
			THEN
				--RAISE NOTICE '	CROSS_ENTROPY Loss';

				-- Calculate softmax of each activity per sample
				WITH
				softmax_top_cte
				AS (SELECT ls.layer_num,
							ls.sample_id, 
							ls.node_num,
							-- Z = SUM(EXP(z - mu))  : normalization
							--    mu = MAX(z)
							-- EQUATION (9) top
							EXP(ls.activity
								- MAX(ls.activity) OVER (PARTITION BY ls.sample_id)) 
								AS exp_result
					FROM tmp_layer_state ls
					WHERE ls.layer_num = lv_num_layers),
				softmax_cte
				AS (SELECT smt.layer_num,
							smt.sample_id, 
							smt.node_num,
							-- Si = Zi / SUM(Z)
							-- EQUATION (9)
							smt.exp_result 
								/ SUM(smt.exp_result) OVER (PARTITION BY smt.sample_id)
								AS sm_result
					FROM softmax_top_cte smt)

				-- Insert softmax outputs to table
				INSERT INTO tmp_delta (layer_num, sample_id, node_num, delta)
				SELECT sm.layer_num,
						sm.sample_id,
						sm.node_num,
						-- EQUATION (14)
						sm.sm_result - CASE WHEN sm.node_num = CAST(sl.label_value AS INT)
												THEN 1		-- ^ where sl.value is a class number [1,2,...C]
											ELSE 0
										END AS delta
				FROM softmax_cte sm
				JOIN tmp_sample_label sl ON sm.sample_id = sl.sample_id;

				--Print cross entropy loss
				--/*
				--FOR iter in
					WITH
					softmax_cte
					AS (SELECT layer_num,
								sample_id,
								node_num,
								delta
						FROM tmp_delta),
					loss_cte
					AS (
						-- Add back 1 from previous insert when calculating loss
						-- Loss = SUM(-1 * LOG(y))	,	y = softmax output of node corresponding to class number
						SELECT -- EQUATION (13) 
							SUM(-1 * LN(CASE WHEN sm.delta+1=0 THEN 1 ELSE sm.delta+1 END)) / COUNT(*) AS avg_loss
						FROM softmax_cte sm
						JOIN tmp_sample_label sl ON sm.sample_id = sl.sample_id
							AND sm.node_num = CAST(sl.label_value AS INT))
					
					SELECT lv_epoch_loss + avg_loss   -- accumulate loss across batches to average later
					INTO lv_epoch_loss
					FROM loss_cte;
				--LOOP
				--	RAISE NOTICE '	Epoch % Batch %: Avg CE Loss = %', lv_epoch_num, lv_batch_num, iter.avg_loss;
				--END LOOP;
				--*/
				
			ELSE
				RAISE NOTICE 'Unknown Loss Function';
			END IF;
			
--RAISE NOTICE '	End Out Deltas: %', (clock_timestamp()::time - lv_last_epoch_start_time);

			-- Insert hidden layer deltas
			FOR lv_layer_iter IN
				SELECT generate_series(lv_num_layers, 2, -1) AS layer_num
			LOOP

				WITH
				lhs_cte 
				AS (SELECT nn.layer_num,
							d.sample_id,
							nn.node_num,
							-- EQUATION (4) (left hand side)
							SUM(d.delta * nn.weight) AS delta_dot_weight
					FROM tmp_delta d
					JOIN neural_network nn ON d.layer_num = nn.layer_num
						AND d.node_num = nn.weight_num
					WHERE nn.network_id = v_network_id 
						AND d.layer_num = lv_layer_iter.layer_num   -- use iter layer num
					GROUP BY nn.layer_num, d.sample_id, nn.node_num),
				delta_cte
				AS (SELECT ls.layer_num, 
							ls.sample_id, 
							ls.node_num,
							-- EQUATION (4)
							lhs.delta_dot_weight
								* CASE WHEN lv_layer_iter.layer_num = 2
									THEN   -- Hidden layer activations
										CASE WHEN lv_output_activation_fn = 'LINEAR'
												THEN 1--result  -- EQUATION (18)
											 WHEN lv_output_activation_fn = 'TANH'
												THEN fn_tanh_prime(ls.result)  -- EQUATION (22)
											 WHEN lv_output_activation_fn = 'RELU'
												THEN fn_relu_prime(ls.result)  -- EQUATION (24)
											 WHEN lv_output_activation_fn = 'SIGMOID'
												THEN fn_sigmoid_prime(ls.result)  -- EQUATION (20)
											 ELSE 1--result
										END 
									ELSE   -- Input layer activations
										CASE WHEN lv_input_activation_fn = 'LINEAR'
												THEN 1--result  -- EQUATION (18)
											 WHEN lv_input_activation_fn = 'TANH'
												THEN fn_tanh_prime(ls.result)  -- EQUATION (22)
											 WHEN lv_input_activation_fn = 'RELU'
												THEN fn_relu_prime(ls.result)  -- EQUATION (24)
											 WHEN lv_input_activation_fn = 'SIGMOID'
												THEN fn_sigmoid_prime(ls.result)  -- EQUATION (20)
											 ELSE 1--result
										END
									END AS delta
					FROM lhs_cte lhs,
						(SELECT layer_num,
						  			sample_id,
						  			node_num,
						  			result,
						  			activity
						  FROM tmp_layer_state ls

						  UNION ALL
					  
						  -- Append 1-valued inputs to match with bias
						  SELECT layer_num,
									sample_id, 
									MAX(node_num) + 1,   -- extra node in next layer
									1,    -- result
									1    -- activity
						  FROM tmp_layer_state ls
						  GROUP BY layer_num, sample_id) ls
					WHERE lhs.layer_num = ls.layer_num + 1 
						AND lhs.sample_id = ls.sample_id 
						AND lhs.node_num = ls.node_num)
						
				INSERT INTO tmp_delta (layer_num, sample_id, node_num, delta)
				SELECT layer_num,
						sample_id,
						node_num,
						delta
				FROM delta_cte;

			END LOOP;   -- end backpropagation layer loop

--RAISE NOTICE '	End Hidden Deltas: %', (clock_timestamp()::time - lv_last_epoch_start_time);

			-- Calculate gradients of weights and biases
			--raise notice 'Gradients';
			--for lv_layer_iter in
			WITH
			layer_cte
			AS (SELECT layer_num,
						sample_id,
						node_num,
						activity
				FROM tmp_layer_state
			 
				UNION ALL
			   	-- Augment layer results with inputs
				SELECT 0,   -- layer_num
						sample_id,
						feature_num,   -- node_num
						feature_value   -- activity
				FROM tmp_sample_feature
				WHERE batch_num = lv_batch_num),
			gradient_cte
			AS (SELECT d.layer_num,
						d.node_num AS d_node_num,
						ls.node_num AS ls_node_num,
						  -- EQUATION (6)
						SUM(d.delta * ls.activity) AS gradient
				FROM tmp_delta d,
					(SELECT layer_num,
								sample_id,
								node_num,
								activity
					FROM layer_cte

					UNION ALL
					
				  	-- Append 1-valued inputs to match with bias
					SELECT layer_num,
							sample_id,
							MAX(node_num) + 1,
							1
					FROM layer_cte
					GROUP BY layer_num, sample_id) ls
				WHERE d.layer_num = ls.layer_num + 1
					AND d.sample_id = ls.sample_id
				GROUP BY d.layer_num, d.node_num, ls.node_num)

			/*select * from gradient_cte
			order by layer_num,ls_node_num,d_node_num
			loop
				raise notice '	L%	N%	W%	%', lv_layer_iter.layer_num,lv_layer_iter.ls_node_num,lv_layer_iter.d_node_num,lv_layer_iter.gradient;
			end loop;*/
			UPDATE neural_network nn
			SET weight = weight - (v_learning_rate / lv_num_samples) * gc.gradient   -- EQUATION (5) (averaged for multiple samples)
			FROM gradient_cte gc
			WHERE nn.network_id = v_network_id	
				AND gc.layer_num = nn.layer_num
				AND gc.ls_node_num = nn.node_num
				AND gc.d_node_num = nn.weight_num;

			-- Clear tables from this batch
			--DELETE FROM tmp_layer_state;
			--DELETE FROM tmp_delta;
			TRUNCATE tmp_layer_state;
			TRUNCATE tmp_delta;
			
--RAISE NOTICE '	End Backward: %', (clock_timestamp()::time - lv_last_epoch_start_time);
		
		END LOOP;   -- End batch loop

		RAISE NOTICE 'Epoch % Avg Loss (%)', lv_epoch_num, lv_epoch_loss/lv_num_batches;
		SELECT 0.0 INTO lv_epoch_loss;
		
	END LOOP;   -- End epoch loop

	-- Drop temp tables
	DROP TABLE tmp_sample_feature;
	DROP TABLE tmp_sample_label;
	DROP TABLE tmp_layer_state;
	DROP TABLE tmp_delta;
	
	RAISE NOTICE 'End (%). Duration (%)', clock_timestamp()::time, clock_timestamp()::time - lv_start_time;
	
END;
$$;


--
-- TOC entry 862 (class 1255 OID 16586)
-- Name: mult(double precision); Type: AGGREGATE; Schema: public; Owner: -
--

CREATE AGGREGATE public.mult(double precision) (
    SFUNC = float8mul,
    STYPE = double precision
);


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 252 (class 1259 OID 19830)
-- Name: airfoil; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.airfoil (
    frequency double precision,
    angle double precision,
    chord_length double precision,
    velocity double precision,
    thickness double precision,
    pressure double precision
);


--
-- TOC entry 223 (class 1259 OID 16587)
-- Name: car; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.car (
    kilometers double precision,
    fuel integer,
    age double precision,
    price double precision
);


--
-- TOC entry 224 (class 1259 OID 16590)
-- Name: car_eval; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.car_eval (
    buy_price text,
    maint_price text,
    doors text,
    capacity text,
    cargo_capacity text,
    safety text,
    acceptability text
);


--
-- TOC entry 225 (class 1259 OID 16596)
-- Name: dataset; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dataset (
    id integer NOT NULL,
    name character varying(30) NOT NULL,
    feature_mean double precision[],
    feature_std double precision[],
    label_min double precision[],
    label_max double precision[],
    normalize_features text,
    normalize_labels text,
    feature_min double precision[],
    feature_max double precision[],
    label_std double precision[],
    label_mean double precision[]
);


--
-- TOC entry 226 (class 1259 OID 16602)
-- Name: iris; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.iris (
    id integer NOT NULL,
    sepal_length numeric(8,2) NOT NULL,
    sepal_width numeric(8,2) NOT NULL,
    petal_length numeric(8,2) NOT NULL,
    petal_width numeric(8,2) NOT NULL,
    species character varying(30),
    for_training boolean DEFAULT true NOT NULL
);


--
-- TOC entry 227 (class 1259 OID 16606)
-- Name: iris_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.iris ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.iris_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 228 (class 1259 OID 16608)
-- Name: log_table; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.log_table (
    id integer NOT NULL,
    value text
);


--
-- TOC entry 229 (class 1259 OID 16614)
-- Name: log_table_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.log_table ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.log_table_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 230 (class 1259 OID 16619)
-- Name: neural_network; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.neural_network (
    network_id integer NOT NULL,
    layer_num integer NOT NULL,
    node_num integer NOT NULL,
    weight_num integer NOT NULL,
    weight double precision NOT NULL
);


--
-- TOC entry 231 (class 1259 OID 16622)
-- Name: neural_network_architecture; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.neural_network_architecture (
    id integer NOT NULL,
    dataset_id integer NOT NULL,
    activation_functions text[] NOT NULL,
    loss_function text NOT NULL,
    name text,
    num_hidden_layers smallint,
    layer_sizes smallint[]
);


--
-- TOC entry 232 (class 1259 OID 16628)
-- Name: neural_network_architecture_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.neural_network_architecture ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.neural_network_architecture_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 233 (class 1259 OID 16630)
-- Name: sample; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sample (
    id integer NOT NULL,
    dataset_id integer NOT NULL,
    features double precision[],
    labels double precision[],
    training boolean NOT NULL
);


--
-- TOC entry 234 (class 1259 OID 16636)
-- Name: sample1; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sample1 (
    id integer NOT NULL,
    training_data_id integer
);


--
-- TOC entry 235 (class 1259 OID 16639)
-- Name: sample_feature; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sample_feature (
    sample_id integer,
    feature_num integer,
    value double precision
);


--
-- TOC entry 236 (class 1259 OID 16642)
-- Name: sample_flat_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.sample ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.sample_flat_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 237 (class 1259 OID 16644)
-- Name: sample_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.sample1 ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.sample_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 238 (class 1259 OID 16646)
-- Name: sample_label; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sample_label (
    sample_id integer,
    label_num integer,
    value character varying(50)
);


--
-- TOC entry 239 (class 1259 OID 16649)
-- Name: sample_test; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sample_test (
    id integer NOT NULL,
    dataset_id integer NOT NULL,
    features double precision[],
    labels double precision[],
    features_norm double precision[],
    labels_norm double precision[],
    training boolean NOT NULL
);


--
-- TOC entry 240 (class 1259 OID 16655)
-- Name: sample_test_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.sample_test ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.sample_test_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 241 (class 1259 OID 16657)
-- Name: test1; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.test1 (
    id integer NOT NULL,
    "values" double precision[]
);


--
-- TOC entry 242 (class 1259 OID 16663)
-- Name: test1_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.test1 ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.test1_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 243 (class 1259 OID 16665)
-- Name: test2; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.test2 (
    id integer NOT NULL,
    index1 integer NOT NULL,
    index2 integer NOT NULL,
    index3 integer NOT NULL,
    value double precision
);


--
-- TOC entry 244 (class 1259 OID 16668)
-- Name: test2_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.test2 ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.test2_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 245 (class 1259 OID 16670)
-- Name: training_data_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.dataset ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.training_data_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 246 (class 1259 OID 16672)
-- Name: vw_active_locks; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.vw_active_locks AS
 SELECT t.schemaname,
    t.relname,
    l.locktype,
    l.page,
    l.virtualtransaction,
    l.pid,
    l.mode,
    l.granted
   FROM (pg_locks l
     JOIN pg_stat_all_tables t ON ((l.relation = t.relid)))
  WHERE ((t.schemaname <> 'pg_toast'::name) AND (t.schemaname <> 'pg_catalog'::name))
  ORDER BY t.schemaname, t.relname;


--
-- TOC entry 247 (class 1259 OID 16677)
-- Name: vw_index_usage; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.vw_index_usage AS
 SELECT pg_stat_user_tables.relname,
    ((100 * pg_stat_user_tables.idx_scan) / (pg_stat_user_tables.seq_scan + pg_stat_user_tables.idx_scan)) AS percent_of_times_index_used,
    pg_stat_user_tables.n_live_tup AS rows_in_table
   FROM pg_stat_user_tables
  WHERE ((pg_stat_user_tables.seq_scan + pg_stat_user_tables.idx_scan) > 0)
  ORDER BY pg_stat_user_tables.n_live_tup DESC;


--
-- TOC entry 248 (class 1259 OID 16686)
-- Name: vw_neural_network; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.vw_neural_network AS
 SELECT neural_network.network_id,
    neural_network.layer_num,
    neural_network.node_num,
    neural_network.weight_num,
    neural_network.weight
   FROM public.neural_network
  ORDER BY neural_network.network_id, neural_network.layer_num, neural_network.node_num, neural_network.weight_num;


--
-- TOC entry 249 (class 1259 OID 16712)
-- Name: vw_table_sizes; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.vw_table_sizes AS
 SELECT pretty_sizes.table_name,
    pg_size_pretty(pretty_sizes.table_size) AS table_size,
    pg_size_pretty(pretty_sizes.indexes_size) AS indexes_size,
    pg_size_pretty(pretty_sizes.total_size) AS total_size,
        CASE
            WHEN (((pretty_sizes.table_type)::text = 'BASE TABLE'::text) AND ((pretty_sizes.table_schema)::text <> ALL (ARRAY[('pg_catalog'::character varying)::text, ('information_schema'::character varying)::text]))) THEN pg_relation_filepath((pretty_sizes.table_name)::regclass)
            ELSE ''::text
        END AS filepath
   FROM ( SELECT all_tables.table_full_name,
            all_tables.table_schema,
            all_tables.table_name,
            pg_table_size((all_tables.table_full_name)::regclass) AS table_size,
            pg_indexes_size((all_tables.table_full_name)::regclass) AS indexes_size,
            pg_total_relation_size((all_tables.table_full_name)::regclass) AS total_size,
            all_tables.table_type
           FROM ( SELECT (((('"'::text || (tables.table_schema)::text) || '"."'::text) || (tables.table_name)::text) || '"'::text) AS table_full_name,
                    tables.table_schema,
                    tables.table_name,
                    tables.table_type
                   FROM information_schema.tables) all_tables
          ORDER BY (pg_total_relation_size((all_tables.table_full_name)::regclass)) DESC) pretty_sizes;


--
-- TOC entry 251 (class 1259 OID 18228)
-- Name: wine_cultivars; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.wine_cultivars (
    cultivars integer NOT NULL,
    alcohol double precision NOT NULL,
    malic_acid double precision NOT NULL,
    ash double precision NOT NULL,
    alcalinity double precision NOT NULL,
    magnesium double precision NOT NULL,
    phenols double precision NOT NULL,
    flavanoids double precision NOT NULL,
    nonflavanoid_phenols double precision NOT NULL,
    proanthocyanins double precision NOT NULL,
    color double precision NOT NULL,
    hue double precision NOT NULL,
    od280_od315 double precision NOT NULL,
    proline double precision NOT NULL
);


--
-- TOC entry 250 (class 1259 OID 16717)
-- Name: wine_quality; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.wine_quality (
    color text,
    fixed_acidity double precision,
    volatile_acidity double precision,
    citric_acid double precision,
    residual_sugar double precision,
    chlorides double precision,
    free_sulfur_dioxide double precision,
    total_sulfur_dioxide double precision,
    density double precision,
    ph double precision,
    sulphates double precision,
    alcohol double precision,
    quality integer
);


--
-- TOC entry 2958 (class 2606 OID 16729)
-- Name: dataset dataset_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset
    ADD CONSTRAINT dataset_pkey PRIMARY KEY (id);


--
-- TOC entry 2962 (class 2606 OID 16731)
-- Name: iris iris_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.iris
    ADD CONSTRAINT iris_pkey PRIMARY KEY (id);


--
-- TOC entry 2967 (class 2606 OID 16733)
-- Name: neural_network_architecture neural_network_architecture_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.neural_network_architecture
    ADD CONSTRAINT neural_network_architecture_pkey PRIMARY KEY (id);


--
-- TOC entry 2965 (class 2606 OID 16735)
-- Name: neural_network pk_neural_network; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.neural_network
    ADD CONSTRAINT pk_neural_network PRIMARY KEY (network_id, layer_num, node_num, weight_num);


--
-- TOC entry 2971 (class 2606 OID 16737)
-- Name: sample sample_flat_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sample
    ADD CONSTRAINT sample_flat_pkey PRIMARY KEY (id);


--
-- TOC entry 2973 (class 2606 OID 16739)
-- Name: sample1 sample_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sample1
    ADD CONSTRAINT sample_pkey PRIMARY KEY (id);


--
-- TOC entry 2975 (class 2606 OID 16741)
-- Name: sample_test sample_test_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sample_test
    ADD CONSTRAINT sample_test_pkey PRIMARY KEY (id);


--
-- TOC entry 2960 (class 2606 OID 16743)
-- Name: dataset unique_dataset_name; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset
    ADD CONSTRAINT unique_dataset_name UNIQUE (name);


--
-- TOC entry 2969 (class 2606 OID 16745)
-- Name: neural_network_architecture uq_neural_network_architecture_name; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.neural_network_architecture
    ADD CONSTRAINT uq_neural_network_architecture_name UNIQUE (name);


--
-- TOC entry 2963 (class 1259 OID 16746)
-- Name: ix_neural_network_net_layer_node; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_neural_network_net_layer_node ON public.neural_network USING btree (network_id, layer_num, node_num);


--
-- TOC entry 2976 (class 2606 OID 16747)
-- Name: neural_network_architecture fk_nna_dataset_id; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.neural_network_architecture
    ADD CONSTRAINT fk_nna_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset(id);


--
-- TOC entry 2979 (class 2606 OID 16752)
-- Name: sample_feature sample_features_sample_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sample_feature
    ADD CONSTRAINT sample_features_sample_id_fkey FOREIGN KEY (sample_id) REFERENCES public.sample1(id);


--
-- TOC entry 2977 (class 2606 OID 16757)
-- Name: sample sample_flat_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sample
    ADD CONSTRAINT sample_flat_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.dataset(id);


--
-- TOC entry 2980 (class 2606 OID 16762)
-- Name: sample_label sample_labels_sample_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sample_label
    ADD CONSTRAINT sample_labels_sample_id_fkey FOREIGN KEY (sample_id) REFERENCES public.sample1(id);


--
-- TOC entry 2981 (class 2606 OID 16767)
-- Name: sample_test sample_test_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sample_test
    ADD CONSTRAINT sample_test_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.dataset(id);


--
-- TOC entry 2978 (class 2606 OID 16772)
-- Name: sample1 sample_training_data_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sample1
    ADD CONSTRAINT sample_training_data_id_fkey FOREIGN KEY (training_data_id) REFERENCES public.dataset(id);


-- Completed on 2020-04-16 10:36:42

--
-- PostgreSQL database dump complete
--

