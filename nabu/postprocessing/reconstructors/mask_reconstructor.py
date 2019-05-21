"""@file mask_reconstructor.py
contains the reconstructor class with use of a mask"""

import reconstructor
from nabu.postprocessing import data_reader
from nabu.processing.feature_computers import base
from abc import ABCMeta, abstractmethod


class MaskReconstructor(reconstructor.Reconstructor):
	"""the general reconstructor class using a mask

	a reconstructor using a mask"""

	__metaclass__ = ABCMeta

	def __init__(self, conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation=False):
		"""MaskReconstructor constructor

		Args:
			conf: the reconstructor configuration as a dictionary
			evalconf: the evaluator configuration as a ConfigParser
			dataconf: the database configuration
			rec_dir: the directory where the reconstructions will be stored
		"""

		super(MaskReconstructor, self).__init__(conf, evalconf, dataconf, rec_dir, task, optimal_frame_permutation)

		self.multi_mic_rec = False
		self.nrmic = 1
		if 'nrmic' in conf and int(conf['nrmic']) > 1:
			self.multi_mic_rec = True
			self.nrmic = int(conf['nrmic'])

		if not self.multi_mic_rec:
			# get the original mixtures reader
			org_mix_names = conf['org_mix'].split(' ')
			org_mix_dataconfs = []
			for org_mix_name in org_mix_names:
				org_mix_dataconfs.append(dict(dataconf.items(org_mix_name)))
			self.org_mix_reader = data_reader.DataReader(org_mix_dataconfs, self.segment_lengths)
		else:
			self.org_mix_readers = []
			for mic_ind in range(self.nrmic):
				org_mix_names = conf['org_mix_%d', (mic_ind+1)].split(' ')
				org_mix_dataconfs = []
				for org_mix_name in org_mix_names:
					org_mix_dataconfs.append(dict(dataconf.items(org_mix_name)))
				self.org_mix_readers.append(data_reader.DataReader(org_mix_dataconfs, self.segment_lengths))

	def reconstruct_signals(self, output):
		"""reconstruct the signals

		Args:
			output: the output of a single utterance of the neural network

		Returns:
			the reconstructed signals
			some info on the utterance"""

		# get the original mixture(s)
		if self.nrmic == 1:
			mixture, utt_info = self.org_mix_reader(self.pos)
		else:
			all_mixtures = []
			all_utt_infos = []
			for mic_ind in range(self.nrmic):
				mixture, utt_info = self.org_mix_readers[mic_ind](self.pos)
				all_mixtures.append(mixture)
				all_utt_infos.append(utt_info)
			# all utt_infos should be the same
			utt_info = all_utt_infos[0]

		# get the masks
		masks = self._get_masks(output, utt_info)

		# apply the masks to obtain the reconstructed signals. Use the conf for feature
		# settings from the original mixture
		if self.nrmic == 1:
			for ind, start_index in enumerate(self.org_mix_reader.start_index_set):
				if start_index > self.pos:
					processor = self.org_mix_reader.processors[ind-1]
					comp_conf = processor.comp.conf
					break

			reconstructed_signals = list()
			for spk in range(self.nrS):
				spec_est = mixture * masks[spk, :, :]
				reconstructed_signals.append(base.spec2time(spec_est, utt_info['rate'], utt_info['siglen'], comp_conf))
		else:
			for ind, start_index in enumerate(self.org_mix_readers[0].start_index_set):
				if start_index > self.pos:
					processor = self.org_mix_readers[0].processors[ind-1]
					comp_conf = processor.comp.conf
					break

			# For each speaker, apply its mask to each microphone signal. The magnitude of the speaker's reconstructed
			# spectrogram is obtained by averaging the masked signals.
			pass

		return reconstructed_signals, utt_info

	def reconstruct_signals_opt_frame_perm(self, output, target):
		"""reconstruct the signals, using optimal frame-level permutations

		Args:
			output: the output of a single utterance of the neural network
			target: the target of a single utterance of the neural network

		Returns:
			the reconstructed signals
			some info on the utterance"""

		# get the original mixture
		mixture, utt_info = self.org_mix_reader(self.pos)

		# get the masks
		masks = self._get_masks_opt_frame_perm(output, target, utt_info)

		# apply the masks to obtain the reconstructed signals. Use the conf for feature
		# settings from the original mixture
		for ind, start_index in enumerate(self.org_mix_reader.start_index_set):
			if start_index > self.pos:
				processor = self.org_mix_reader.processors[ind-1]
				comp_conf = processor.comp.conf
				break

		reconstructed_signals = list()
		for spk in range(self.nrS):
			spec_est = mixture * masks[spk, :, :]
			reconstructed_signals.append(base.spec2time(spec_est, utt_info['rate'], utt_info['siglen'], comp_conf))

		return reconstructed_signals, utt_info

	@abstractmethod
	def _get_masks(self, output, utt_info):
		"""estimate the masks

		Args:
			output: the output of a single utterance of the neural network
			utt_info: some info on the utterance

		Returns:
			the estimated masks"""


