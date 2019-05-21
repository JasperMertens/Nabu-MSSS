import os
import numpy as np
import copy
import math

from skopt.space.space import Categorical, Integer


def check_parameter_count(vals_dict, param_thr, par_cnt_scheme='enc_dec_cnn_lstm_ff'):
	if par_cnt_scheme == 'enc_dec_cnn_lstm_ff':
		if vals_dict['cnn_num_enc_lay'] == 0:
			if vals_dict['lstm_num_lay'] == 0 or vals_dict['concat_flatten_last_2dims_cnn'] == 'False':
				values_suitable = False
				par_cnt_dict = {'total': 0, 'cnn': 0, 'lstm': 0, 'ff': 0}
				return values_suitable, par_cnt_dict

		par_cnt_cnn = 0
		par_cnt_lstm = 0
		par_cnt_ff = 0

		#
		n_lstm = \
			vals_dict['lstm_num_un_lay1'] * \
			(vals_dict['lstm_fac_per_lay'] ** np.arange(vals_dict['lstm_num_lay']))
		n_lstm = [int(np.ceil(n)) for n in n_lstm]
		lstm_bidir = int(vals_dict['lstm_bidir'] == 'dblstm') + 1  # 1 if single direction, 2 if bidirectional

		n_ff = \
			vals_dict['ff_num_un_lay1'] * \
			(vals_dict['ff_fac_per_lay'] ** np.arange(vals_dict['ff_num_lay']))
		n_ff = [int(np.ceil(n)) for n in n_ff]
		concat_flatten = int(vals_dict['concat_flatten_last_2dims_cnn'] == 'True')  # 1 if flatten, else 0

		F = 129  # number of frequency bins
		D = 20  # embedding dimension
		n_out = D * (1 + (F - 1) * concat_flatten)

		kernel_shape = [vals_dict['cnn_kernel_size_t'], vals_dict['cnn_kernel_size_f']]
		kernel_fac_after_pool = [
			vals_dict['cnn_kernel_size_t_fac_after_pool'], vals_dict['cnn_kernel_size_f_fac_after_pool']]
		max_pool_rate = [vals_dict['cnn_max_pool_rate_t'], vals_dict['cnn_max_pool_rate_f']]
		kernel_shapes_enc = [kernel_shape]
		for l_ind in range(vals_dict['cnn_num_enc_lay']):
			kernel_l_plus1 = copy.deepcopy(kernel_shapes_enc[l_ind])
			if np.mod(l_ind + 1, max_pool_rate[0]) == 0:
				kernel_l_plus1[0] *= kernel_fac_after_pool[0]
			if np.mod(l_ind + 1, max_pool_rate[1]) == 0:
				kernel_l_plus1[1] *= kernel_fac_after_pool[1]
			kernel_shapes_enc.append(kernel_l_plus1)
		kernel_shapes_enc = [[int(math.ceil(k)) for k in kern] for kern in kernel_shapes_enc]
		kernel_sizes_enc = [kern[0] * kern[1] for kern in kernel_shapes_enc]
		kernel_sizes_dec = kernel_sizes_enc[::-1]
		kernel_sizes_dec = kernel_sizes_dec[1:]

		cnn_bypass = int(vals_dict['cnn_bypass'] == 'True')  # 1 if bypass, else 0

		#
		n_enc = \
			vals_dict['cnn_num_un_lay1'] * (vals_dict['cnn_fac_per_lay'] ** np.arange(vals_dict['cnn_num_enc_lay']))
		n_enc = [int(np.ceil(n)) for n in n_enc]
		n_dec = n_enc[::-1]
		n_dec = n_dec[1:] + [vals_dict['cnn_num_un_out']]

		# cnn_param_cnt style: kernel_size * n_un * n_inputs + n_un (last term for bias)
		# encoder cnn parameter count
		for l_ind in range(vals_dict['cnn_num_enc_lay']):
			if l_ind == 0:
				par_cnt_cnn += kernel_sizes_enc[l_ind] * (n_enc[l_ind] * 1) + n_enc[l_ind]
			else:
				par_cnt_cnn += kernel_sizes_enc[l_ind] * (n_enc[l_ind] * n_enc[l_ind - 1]) + n_enc[l_ind]

		# decoder cnn parameter count
		for l_ind in range(vals_dict['cnn_num_enc_lay']):
			corr_enc_l_ind = vals_dict['cnn_num_enc_lay'] - 1 - l_ind
			if l_ind == 0:
				par_cnt_cnn += \
					kernel_sizes_enc[corr_enc_l_ind] * (n_dec[l_ind] * (n_enc[corr_enc_l_ind])) + n_dec[l_ind]
			else:
				par_cnt_cnn += \
					kernel_sizes_enc[corr_enc_l_ind] * \
					(n_dec[l_ind] * (n_dec[l_ind - 1] + cnn_bypass * n_enc[corr_enc_l_ind])) + n_dec[l_ind]

		# lstm parameter count
		for l_ind in range(vals_dict['lstm_num_lay']):
			if l_ind == 0:
				par_cnt_lstm += lstm_bidir * (4 * n_lstm[l_ind] * (n_lstm[l_ind] + F + 1))
			else:
				par_cnt_lstm += lstm_bidir * (4 * n_lstm[l_ind] * (n_lstm[l_ind] + lstm_bidir * n_lstm[l_ind - 1] + 1))

		# feedforward and output layer parameter count

		if vals_dict['lstm_num_lay'] == 0:
			outputs_lstm = 0
		else:
			outputs_lstm = lstm_bidir * n_lstm[-1]

		if vals_dict['cnn_num_enc_lay'] == 0:
			outputs_cnn = 0
		else:
			outputs_cnn = n_dec[-1] * (1 + (F - 1) * concat_flatten)

		outputs_lstm_cnn = outputs_lstm + outputs_cnn

		for l_ind in range(vals_dict['ff_num_lay']):
			if l_ind == 0:
				par_cnt_ff += n_ff[l_ind] * outputs_lstm_cnn + n_ff[l_ind]

			else:
				par_cnt_ff += n_ff[l_ind] * n_ff[l_ind - 1] + n_ff[l_ind]

		if vals_dict['ff_num_lay'] > 0:
			par_cnt_ff += n_out * n_ff[-1] + n_out
		else:
			par_cnt_ff += n_out * outputs_lstm_cnn + n_out

		par_cnt = par_cnt_lstm + par_cnt_cnn + par_cnt_ff

		if par_cnt > param_thr or par_cnt < param_thr * (1 - 0.05):
			values_suitable = False
		else:
			values_suitable = True

		par_cnt_dict = {'total': par_cnt, 'cnn': par_cnt_cnn, 'lstm': par_cnt_lstm, 'ff': par_cnt_ff}

	else:
		raise ValueError('Parameter count scheme %s is unknown', par_cnt_scheme)

	return values_suitable, par_cnt_dict


def check_parameter_count_for_sample(dim_values, hyper_param_names, param_thr, par_cnt_scheme='enc_dec_cnn_lstm_ff'):
	if isinstance(dim_values, np.ndarray):
		# dim values is given as a numpy array
		if len(dim_values.shape) > 1:
			# dim values is given as a [n_samples, num_dim] numpy array
			multi_samples = True
			n_samples = dim_values.shape[0]
		else:
			# dim values is given as a [num_dim] numpy array
			multi_samples = False
	else:
		# dim values is given as a list
		multi_samples = any(isinstance(el, list) for el in dim_values)
		if multi_samples:
			# dim values is given as a [n_samples, num_dim] list
			n_samples = len(dim_values)

	if not multi_samples:
		vals_dict = {name: val for (name, val) in zip(hyper_param_names, dim_values)}

		return check_parameter_count(vals_dict, param_thr, par_cnt_scheme)
	else:
		all_values_suitable = []
		all_par_cnt_dict = []
		for dim_value_sample in dim_values:
			vals_dict = {name: val for (name, val) in zip(hyper_param_names, dim_value_sample)}

			values_suitable, par_cnt_dict = check_parameter_count(vals_dict, param_thr, par_cnt_scheme)
			all_values_suitable.append(values_suitable)
			all_par_cnt_dict.append(par_cnt_dict)

		return all_values_suitable, all_par_cnt_dict


def partial_dependence_valid_samples(
		space, model, param_thr, hyper_param_names, i, j=None, par_cnt_scheme='enc_dec_cnn_lstm_ff', sample_points=None,
		n_samples=250, n_points=40):
	"""Copied from skopts. Calculate the partial dependence for dimensions `i` and `j` with
	respect to the objective value, as approximated by `model`. Only consider samples that are valid according to
	param_thr and par_cnt_scheme

	The partial dependence plot shows how the value of the dimensions
	`i` and `j` influence the `model` predictions after "averaging out"
	the influence of all other dimensions.

	Parameters
	----------
	* `space` [`Space`]
		The parameter space over which the minimization was performed.

	* `model`
		Surrogate model for the objective function.

	* `param_thr` [int]
		threshold on trainable parameter count

	* `hyper_param_names`
		Names of the hyper parameters

	* `i` [int]
		The first dimension for which to calculate the partial dependence.

	* `j` [int, default=None]
		The second dimension for which to calculate the partial dependence.
		To calculate the 1D partial dependence on `i` alone set `j=None`.

	* `par_cnt_scheme` [default='enc_dec_cnn_lstm_ff']
		The scheme to use to count the trainable parameters, to check if a sample is valid.

	* `sample_points` [np.array, shape=(n_points, n_dims), default=None]
		Randomly sampled and transformed points to use when averaging
		the model function at each of the `n_points`. These should already be valid samples.

	* `n_samples` [int, default=100]
		Number of random samples to use for averaging the model function
		at each of the `n_points`. Only used when `sample_points=None`.

	* `n_points` [int, default=40]
		Number of points at which to evaluate the partial dependence
		along each dimension `i` and `j`.

	Returns
	-------
	For 1D partial dependence:

	* `xi`: [np.array]:
		The points at which the partial dependence was evaluated.

	* `yi`: [np.array]:
		The value of the model at each point `xi`.

	For 2D partial dependence:

	* `xi`: [np.array, shape=n_points]:
		The points at which the partial dependence was evaluated.
	* `yi`: [np.array, shape=n_points]:
		The points at which the partial dependence was evaluated.
	* `zi`: [np.array, shape=(n_points, n_points)]:
		The value of the model at each point `(xi, yi)`.
	"""
	if sample_points is None:
		sample_points = space.rvs(n_samples=n_samples)
		suitable_sample_points, _ = check_parameter_count_for_sample(
			sample_points, hyper_param_names, param_thr, par_cnt_scheme)

		sample_points = [sample_points[ind] for ind, suit in enumerate(suitable_sample_points) if suit]

		sample_points = space.transform(sample_points)

	if j is None:
		bounds = space.dimensions[i].bounds
		# XXX use linspace(*bounds, n_points) after python2 support ends
		n_points_i = n_points
		if isinstance(space.dimensions[i], Integer) and (bounds[1] - bounds[0] + 1) < n_points:
			n_points_i = bounds[1] - bounds[0] + 1
		xi = np.linspace(bounds[0], bounds[1], n_points_i)
		xi_transformed = space.dimensions[i].transform(xi)

		yi = []
		for x_ in xi_transformed:
			rvs_ = np.array(sample_points)
			rvs_[:, i] = x_
			tmp = space.inverse_transform(rvs_)
			suitable_points, _ = check_parameter_count_for_sample(tmp, hyper_param_names, param_thr, par_cnt_scheme)
			rvs_ = [rvs_[ind] for ind, suit in enumerate(suitable_points) if suit]
			if len(rvs_) < 5:
				print \
					'WARNING: Not sufficient samples available to predict partial dependence for value (%.3f). ' \
					'Setting it to 0.175' % x_
				yi.append(0.175)
				continue
			elif len(rvs_) < 20:
				print \
					'WARNING: only %d value(s) used to predict partial dependence for value (%.3f)' % \
					(len(rvs_), x_)

			yi.append(np.mean(model.predict(rvs_)))

		return xi, yi

	else:
		bounds = space.dimensions[j].bounds
		# XXX use linspace(*bounds, n_points) after python2 support ends
		n_points_j = n_points
		if isinstance(space.dimensions[j], Integer) and (bounds[1] - bounds[0] + 1) < n_points:
			n_points_j = bounds[1] - bounds[0] + 1
		xi = np.linspace(bounds[0], bounds[1], n_points_j)
		xi_transformed = space.dimensions[j].transform(xi)

		bounds = space.dimensions[i].bounds
		# XXX use linspace(*bounds, n_points) after python2 support ends
		n_points_i = n_points
		if isinstance(space.dimensions[i], Integer) and (bounds[1] - bounds[0] + 1) < n_points:
			n_points_i = bounds[1] - bounds[0] + 1
		yi = np.linspace(bounds[0], bounds[1], n_points_i)
		yi_transformed = space.dimensions[i].transform(yi)

		zi = []
		for x_ in xi_transformed:
			row = []
			for y_ in yi_transformed:
				rvs_ = np.array(sample_points)
				rvs_[:, (j, i)] = (x_, y_)
				tmp = space.inverse_transform(rvs_)
				suitable_points, _ = check_parameter_count_for_sample(tmp, hyper_param_names, param_thr, par_cnt_scheme)
				rvs_ = [rvs_[ind] for ind, suit in enumerate(suitable_points) if suit]
				if len(rvs_) < 5:
					print \
						'WARNING: Not sufficient samples available to predict partial dependence for values (%.3f, %.3f). ' \
						'Setting it to 0.175' % (x_, y_)
					row.append(0.175)
					continue
				elif len(rvs_) < 20:
					print \
						'WARNING: only %d value(s) used to predict partial dependence for values (%.3f, %.3f)' % \
						(len(rvs_), x_, y_)
				row.append(np.mean(model.predict(rvs_)))
			zi.append(row)

		return xi, yi, np.array(zi).T


def partial_dependence_valid_samples_allow_paramcounts(
		space, model, param_thr, hyper_param_names, i, j=None, par_cnt_scheme='enc_dec_cnn_lstm_ff', sample_points=None,
		n_samples=250, n_points=40):
	"""Copied from skopts. Calculate the partial dependence for dimensions `i` and `j` with
	respect to the objective value, as approximated by `model`. Only consider samples that are valid according to
	param_thr and par_cnt_scheme. i and j may also be strings that direct to parameter counts

	The partial dependence plot shows how the value of the dimensions
	`i` and `j` influence the `model` predictions after "averaging out"
	the influence of all other dimensions.

	Parameters
	----------
	* `space` [`Space`]
		The parameter space over which the minimization was performed.

	* `model`
		Surrogate model for the objective function.

	* `param_thr` [int]
		threshold on trainable parameter count

	* `hyper_param_names`
		Names of the hyper parameters

	* `i` [int]
		The first dimension for which to calculate the partial dependence.

	* `j` [int, default=None]
		The second dimension for which to calculate the partial dependence.
		To calculate the 1D partial dependence on `i` alone set `j=None`.

	* `par_cnt_scheme` [default='enc_dec_cnn_lstm_ff']
		The scheme to use to count the trainable parameters, to check if a sample is valid.

	* `sample_points` [np.array, shape=(n_points, n_dims), default=None]
		Randomly sampled and transformed points to use when averaging
		the model function at each of the `n_points`. These should already be valid samples.

	* `n_samples` [int, default=100]
		Number of random samples to use for averaging the model function
		at each of the `n_points`. Only used when `sample_points=None`.

	* `n_points` [int, default=40]
		Number of points at which to evaluate the partial dependence
		along each dimension `i` and `j`.

	Returns
	-------
	For 1D partial dependence:

	* `xi`: [np.array]:
		The points at which the partial dependence was evaluated.

	* `yi`: [np.array]:
		The value of the model at each point `xi`.

	For 2D partial dependence:


	* `xi`: [np.array, shape=n_points]:
		The points at which the partial dependence was evaluated.
	* `yi`: [np.array, shape=n_points]:
		The points at which the partial dependence was evaluated.
	* `zi`: [np.array, shape=(n_points, n_points)]:
		The value of the model at each point `(xi, yi)`.
	"""
	if sample_points is None:
		sample_points = space.rvs(n_samples=n_samples)
		suitable_sample_points, all_param_dicts = check_parameter_count_for_sample(
			sample_points, hyper_param_names, param_thr, par_cnt_scheme)

		sample_points = [sample_points[ind] for ind, suit in enumerate(suitable_sample_points) if suit]
		suitable_param_dicts = [all_param_dicts[ind] for ind, suit in enumerate(suitable_sample_points) if suit]

		sample_points = space.transform(sample_points)

	if j is None:
		yi = []
		yi_std = []
		if isinstance(i, str):
			# using parameter counts.
			i_param_cnt = True
			# assign every sample to a 'cluster'
			bins = np.linspace(-1, param_thr, n_points+1)
			bin_width = bins[1]-bins[0]
			xi = bins[:-1]
			xi = [int(xii+bin_width/2) for xii in xi]
			xi_transformed = bins[:-1]
			params_i = [param_dict[i] for param_dict in suitable_param_dicts]
			cluster_inds = np.digitize(params_i, bins, right=True)
			cluster_inds = [ind-1 for ind in cluster_inds]
		else:
			i_param_cnt = False
			bounds = space.dimensions[i].bounds
			# XXX use linspace(*bounds, n_points) after python2 support ends
			n_points_i = n_points
			if isinstance(space.dimensions[i], Integer) and (bounds[1] - bounds[0] + 1) < n_points:
				n_points_i = bounds[1] - bounds[0] + 1
			xi = np.linspace(bounds[0], bounds[1], n_points_i)
			xi_transformed = space.dimensions[i].transform(xi)

		for x_ind, x_ in enumerate(xi_transformed):
			if i_param_cnt:
				rvs_ = np.array([
					sample_points[ind] for ind, cluster_ind in enumerate(cluster_inds) if cluster_ind == x_ind])
			else:
				rvs_ = np.array(sample_points)
				rvs_[:, i] = x_

			if len(rvs_) > 0:
				tmp = space.inverse_transform(rvs_)
				suitable_points, _ = check_parameter_count_for_sample(tmp, hyper_param_names, param_thr, par_cnt_scheme)
			else:
				suitable_points = []
			rvs_ = [rvs_[ind] for ind, suit in enumerate(suitable_points) if suit]
			if len(rvs_) < 5:
				print \
					'WARNING: Not sufficient samples available to predict partial dependence for value (%.3f). ' \
					'Setting it to 0.175' % x_
				yi.append(0.175)
				yi_std.append(0)
				continue
			elif len(rvs_) < 20:
				print \
					'WARNING: only %d value(s) used to predict partial dependence for value (%.3f)' % \
					(len(rvs_), x_)

			predictions = model.predict(rvs_)
			yi.append(np.mean(predictions))
			yi_std.append(np.std(predictions))

		return xi, np.array(yi), np.array(yi_std)

	else:
		zi = []

		if isinstance(j, str):
			# using parameter counts.
			x_param_cnt = True
			# assign every sample to a 'cluster'
			bins_x = np.linspace(-1, param_thr, n_points)
			bin_width = bins_x[1]-bins_x[0]
			xi = bins_x[:-1]
			xi = [int(xii+bin_width/2) for xii in xi]
			xi_transformed = bins_x[:-1]
			params_j = [param_dict[j] for param_dict in suitable_param_dicts]
		else:
			x_param_cnt = False
			bounds = space.dimensions[j].bounds
			# XXX use linspace(*bounds, n_points) after python2 support ends
			n_points_j = n_points
			if isinstance(space.dimensions[j], Integer) and (bounds[1] - bounds[0] + 1) < n_points:
				n_points_j = bounds[1] - bounds[0] + 1
			xi = np.linspace(bounds[0], bounds[1], n_points_j)
			xi_transformed = space.dimensions[j].transform(xi)

		if isinstance(i, str):
			# using parameter counts.
			y_param_cnt = True
			# assign every sample to a 'cluster'
			bins_y = np.linspace(-1, param_thr, n_points)
			bin_width = bins_y[1]-bins_y[0]
			yi = bins_y[:-1]
			yi = [int(yii+bin_width/2) for yii in yi]
			yi_transformed = bins_y[:-1]
			params_i = [param_dict[i] for param_dict in suitable_param_dicts]
		else:
			y_param_cnt = False
			bounds = space.dimensions[i].bounds
			# XXX use linspace(*bounds, n_points) after python2 support ends
			n_points_i = n_points
			if isinstance(space.dimensions[i], Integer) and (bounds[1] - bounds[0] + 1) < n_points:
				n_points_i = bounds[1] - bounds[0] + 1
			yi = np.linspace(bounds[0], bounds[1], n_points_i)
			yi_transformed = space.dimensions[i].transform(yi)

		for x_ind, x_ in enumerate(xi_transformed):
			row = []
			for y_ind, y_ in enumerate(yi_transformed):
				rvs_ = np.array(sample_points)
				if not x_param_cnt:
					rvs_[:, j] = x_
				if not y_param_cnt:
					rvs_[:, i] = y_

				if len(rvs_) > 0:
					tmp = space.inverse_transform(rvs_)
					suitable_points, param_dicts = check_parameter_count_for_sample(
						tmp, hyper_param_names, param_thr, par_cnt_scheme)
				else:
					suitable_points = []
					param_dicts = []
				rvs_ = [rvs_[ind] for ind, suit in enumerate(suitable_points) if suit]
				suitable_param_dicts_i_j = [param_dicts[ind] for ind, suit in enumerate(suitable_points) if suit]

				if x_param_cnt:
					# only select samples that are in the correct x_bin
					params_i_i_j = [param_dict[j] for param_dict in suitable_param_dicts_i_j]
					cluster_inds_i_j = np.digitize(params_i_i_j, bins_x, right=True)
					cluster_inds_i_j = [ind-1 for ind in cluster_inds_i_j]
					inds_to_retain = [
						ind for ind, cluster_ind in enumerate(cluster_inds_i_j) if cluster_ind == x_ind]
					rvs_ = np.array([rvs_[ind] for ind in inds_to_retain])
					suitable_param_dicts_i_j = np.array([suitable_param_dicts_i_j[ind] for ind in inds_to_retain])

				if y_param_cnt:
					# only select samples that are in the correct x_bin
					params_i_i_j = [param_dict[i] for param_dict in suitable_param_dicts_i_j]
					cluster_inds_i_j = np.digitize(params_i_i_j, bins_y, right=True)
					cluster_inds_i_j = [ind-1 for ind in cluster_inds_i_j]
					inds_to_retain = [
						ind for ind, cluster_ind in enumerate(cluster_inds_i_j) if cluster_ind == y_ind]
					rvs_ = np.array([rvs_[ind] for ind in inds_to_retain])
					suitable_param_dicts_i_j = np.array([suitable_param_dicts_i_j[ind] for ind in inds_to_retain])

				if len(rvs_) < 5:
					print \
						'WARNING: Not sufficient samples available to predict partial dependence for values (%.3f, %.3f). ' \
						'Setting it to 0.175' % (x_, y_)
					row.append(0.175)
					continue
				elif len(rvs_) < 20:
					print \
						'WARNING: only %d value(s) used to predict partial dependence for values (%.3f, %.3f)' % \
						(len(rvs_), x_, y_)
				row.append(np.mean(model.predict(rvs_)))
			zi.append(row)

		return xi, yi, np.array(zi).T


def distance(space, point_a, point_b):
	"""Compute distance between two points in this space.

	Parameters
	----------
	* `space` [skopt.space object]
		Space in wich to calculate the distance

	* `a` [array]
		First point.

	* `b` [array]
		Second point.
	"""
	tot_distance = 0.
	for a, b, dim in zip(point_a, point_b, space.dimensions):
		if isinstance(dim, Categorical):
			tot_distance += dim.distance(a, b)
		else:
			a_transf = dim.transform(a)
			b_transf = dim.transform(b)
			tot_distance += abs(b_transf - a_transf)

	return tot_distance/len(point_a)


excpected_condor_string = 'Submitting job(s).\n1 job(s) submitted to cluster '


def run_job(job_string, computing):
	job_output = os.popen(job_string).read()
	print job_output
	if computing == 'condor':
		if len(job_output) > len(excpected_condor_string) and \
				job_output[:len(excpected_condor_string)] == excpected_condor_string:
			job_id = job_output[len(excpected_condor_string):-2]

		else:
			raise ValueError('WARNING: Could not parse process output. Possibly, an error occured')

	else:
		job_id = '0'

	return job_id

