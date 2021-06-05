#!/usr/bin/env python3

import os, sys
import numpy as np
#from numba import jit
import librosa
from scipy import ndimage
import sounddevice as sd
from scipy.io.wavfile import write
from matplotlib import pyplot as plt
import time
import codecs, json 
import asyncio
import queue
import sys
import soundfile as sf
import pickle
import tkinter
import profile
import operator


from albumData import albums as albums


class TheMilkcrate():
	def __init__(self):
		# Tolerance parameters
		self.dist_freq = 13  # kappa: neighborhood in frequency direction
		self.dist_time = 5   # tau: neighborhood in time direction
		
		self.N = 2**13 # number 
		self.H = int(self.N/2)
		self.bin_max= 2**8
		
		## FOR CONTROLLING WHAT SPECIFIC FREQUENCIES WE ARE MAPPING
		freq_delta_bin = (44100/self.N) #delta between bins
		upper_bound    = 4000/freq_delta_bin #limit it to 8kHz
		self.lower_bound    = int(np.floor(50/freq_delta_bin)) #limit it to what my speakers can produce
		self.jump           = int(np.floor(upper_bound/self.bin_max)) #how many bins are in a jump
		self.hop_bound = self.bin_max*self.jump+self.lower_bound #upper bound of hop

		self.milkcrate = {}
		self.load_pickle()
				
		self.RATE = 44100
		self.DURATION = 6
		self.CHUNK = self.RATE * self.DURATION
		
		self.delta_threshold = 10

	########################################################################
	#############   FUNCTIONS RELATED TO FILE MANIPULATION #################
	########################################################################


	def load_wav(self, fn_wav, Fs):
		""" load_wav

		Args:
		    fn_wav: audio sample in question
		    Fs: sampling frequency

		Returns:
		    x: audio sample as a numpy array
		    Fs: sampling frequency
		"""
		x, Fs = librosa.load(fn_wav, Fs)
		x = x.astype(np.float32)
		return (x,Fs)

	def save_pickle(self):
		milkcrate_copy = self.milkcrate.copy()
		os.remove('milkcrate.p')
		pickle.dump(milkcrate_copy, open("milkcrate.p", 'wb'))		
		print('The Milk Crate saved to pickle . . .')

	def load_pickle(self):
		self.milkcrate = pickle.load(open("milkcrate.p", 'rb'))
		print('The Milk Crate loaded from pickle . . .')


	########################################################################
	############# FUNCTIONS RELATED TO COMPUTATION OF CMAP #################
	########################################################################

	def compute_spectrogram(self, x, Fs=44100):
		""" Compute Spectrogram

		Args:
		    fn_wav: audio sample in question
		    Fs: sampling frequency
		    N: window length
		    H: hop size
			bin_max: frequency bins
			frame_max: number of time windows

		Returns:
		    Y: bin_max by frame_max spectrogram
		"""
		X = librosa.stft(x, n_fft=self.N, hop_length=self.H, win_length=self.N, window='hanning')
		X = ndimage.maximum_filter(np.abs(X), size=[self.jump+1, 1], mode='constant')
		Y = X[self.lower_bound:self.hop_bound:self.jump,:]

		return Y

	def compute_constellation_map(self, Y, thresh=0.01):
		"""Compute constellation map

		Args:
		    Y: Spectrogram (magnitude)
		    dist_freq: Neighborhood parameter for frequency direction (kappa)
		    dist_time: Neighborhood parameter for time direction (tau)
		    thresh: Threshold parameter for minimal peak magnitude

		Returns:
		    Cmap: Boolean mask for peak structure (same size as Y)
		"""
		# spectrogram dimensions
		result = ndimage.maximum_filter(Y, size=[2*self.dist_freq+1, 2*self.dist_time+1], mode='constant')
		Cmap = np.logical_and(Y == result, result > thresh)
		return Cmap


	def h(self, mat):
		"""hash function (naive implementation)

		Args:
		    mat: cmap matrix

		Returns:
		    h: hashed dictionary
		"""
		(rows, cols) = mat.shape
		h = {}
		for row in range(rows):
			tmp = np.where(mat[row] == True)[0]
			if tmp.shape[0] != 0:
				h[row] = tmp
		return h


	def generate_cmap(self, indata, Fs = 44100, **kwargs):
		Y_Q = self.compute_spectrogram(indata[:,0], Fs)
		Cmap_Q = self.compute_constellation_map(Y_Q)
		return Cmap_Q



	########################################################################
	############# FUNCTIONS RELATED TO MATCHING OF CMAP    #################
	########################################################################

	def hash_matching_function(self, CD_h, nd, CQ_h, nq):
		tmp  = [0]*(nd+nq*2)
		for key in CQ_h.keys():
			if key not in CD_h: continue
			CQ_h_list = CQ_h[key]
			CD_h_list = CD_h[key]
			for n_val in CQ_h_list:
				vals = CD_h[key] - n_val + nq
				for val in vals: tmp[val] += 1
		delta     = max(tmp)
		shift_max = tmp.index(delta) - nq
		return delta, shift_max

	def hash_match_lookup(self, Cmap_Q):
		delta_tmp = 0
		shift_tmp = 0
		CQ_h = self.h(Cmap_Q)
		(kq, nq) = Cmap_Q.shape
		for album in self.milkcrate:
			if album == 999999: continue
			for song in self.milkcrate[album]['songs']:
				CD_h = self.milkcrate[album]['songs'][song]['hash_cmap']
				(kd, nd) = self.milkcrate[album]['songs'][song]['cmap'].shape
				(delta, shift_max) = self.hash_matching_function(CD_h, nd, CQ_h, nq)
				if delta > delta_tmp:
					shift_tmp = shift_max
					delta_tmp = delta
					max_delta = (song, album)
		return (max_delta, delta_tmp, shift_tmp)


	def match_the_wax(self, indata, **kwargs):
		Cmap_Q = self.generate_cmap(indata, Fs=44100)
		print('--HASH LOOKUP--')	
		start_time = time.time()
		(max_delta,delta, shift_max) = self.hash_match_lookup(Cmap_Q)
		end_time = time.time() 
		hash_time = end_time-start_time
		print('Total Time: %.4f' %(hash_time))
		(song, album) = max_delta
		max_delta = '%s by %s on %s' % (self.milkcrate[album]['songs'][song]['title'], self.milkcrate[album]['artist'], self.milkcrate[album]['album'])
		if delta > self.delta_threshold:
			flag = True
		else:
			flag = False
		return (max_delta, delta, shift_max, flag)

#########################################
#### CLASS FOR HANDLING MICROPHONE ######
#########################################
class AudioHandler(object):
	def __init__(self):
		self.CHANNELS = 1
		self.RATE = 44100
		self.stream = None
		self.DURATION = 6
		self.CHUNK = self.RATE * self.DURATION
		self.counter = 0


	async def inputstream_generator(self, channels=1, samplerate=44100, blocksize=88200, **kwargs):
		"""Generator that yields blocks of input data as NumPy arrays."""
		
		blocksize= self.CHUNK
		q_in = asyncio.Queue() #create a queue to draw from
		loop = asyncio.get_event_loop() #this is the loop that will continously add to the queue

		def callback(indata, frame_count, time_info, status):
			loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

		stream = sd.InputStream(callback=callback, channels=channels, samplerate=samplerate, blocksize=blocksize, **kwargs)

		with stream:
			while True:
				indata, status = await q_in.get()
				yield indata, status


	async def do_func_2_stream(self, func, **kwargs):
		"""Show minimum and maximum value of each incoming audio block."""
		async for indata, status in self.inputstream_generator(**kwargs):
			if status:
				print(status)
			(max_delta, delta, shift_max, flag) = func(indata, **kwargs)
		return (max_delta, delta, shift_max, flag)


async def Run_MTW(**kwargs):
	tmc  = TheMilkcrate()
	audio = AudioHandler()
	
	try:
	    await asyncio.wait_for(audio.do_func_2_stream(tmc.match_the_wax, channels=audio.CHANNELS, samplerate=audio.RATE, blocksize=audio.CHUNK, **kwargs), timeout=120)
	except asyncio.TimeoutError:
	    pass
	
	print('That\'s all folks!')


async def main():
    # Schedule three calls *concurrently*:
    await asyncio.gather(
        Run_MTW()
    )



def test_run(**kwargs):
	tmc  = TheMilkcrate()

	num_songs = 0
	num_albums = 0
	tot_time = 0
	for album in tmc.milkcrate:
		num_albums += 1
		if album != '999999':
			for song in tmc.milkcrate[album]['songs']:
				num_songs += 1
				tot_time += tmc.milkcrate[album]['songs'][song]['track_length']


	print('\n----Milkcrate Stats----')
	hours = int(tot_time/(60*60))
	remain = tot_time%(hours*60*60)
	mins = int(remain/60)
	seconds = remain%60
	print('Albums in the Milkcrate: %d' % num_albums)
	print('Songs in the Milkcrate: %d' % num_songs)
	print('Time in the Milkcrate: %d:%d:%d' % (hours,mins,seconds))

	recordings = []
	for i in range(5,11):
		key = 'A%d'%i 
		recordings.append(tmc.milkcrate[999999]['songs'][key])

	for song in recordings:
		print('\n#####   %s   #####' % song['title'])
		Cmap_Q_rec = song['cmap']

		print('--HASH LOOKUP--')	
		start_time = time.time()
		(max_delta,delta, shift_max) = tmc.hash_match_lookup(Cmap_Q_rec)
		end_time = time.time() 
		hash_time = end_time-start_time
		print('Total Time: %.4f' %(hash_time))
		print('Avg.  Time: %.2fms' % ((hash_time)/num_songs*1000))
		(song, album) = max_delta
		max_delta = '%s by %s on %s' % (tmc.milkcrate[album]['songs'][song]['title'], tmc.milkcrate[album]['artist'], tmc.milkcrate[album]['album'])
		if delta > tmc.delta_threshold:
			print(max_delta, delta, shift_max)
		else:
			print('No match could be found . . .')



if __name__ == "__main__":
	# test_run()

	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		sys.exit('\nInterrupted by user')