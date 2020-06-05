'''
    Read and inspect the data provided by the supervisors (convert raw data to spectograms).
    Describe it in the methods sections. Describe your strategy of tackling the problem.

    Todo:
        - take the fft on parts of size 1 mio datapoints, not a whole recording at once
        - batch processing


     There is an online documentation of the data creation tool:
       Link is in the presentation (try to get the newest version).
     https://www.authorea.com/users/53192/articles/321045-sdr-birdrec-software-documentation

     For all visualizations:
       Make sure you look at at least ~1 second of recording.

    ** What vocalizations, etc look like **
     Radio noise:
          - only in the backpack recordings;
          - it really looks like noise and covers the whole spectrum
     Wing flaps:
          - Vertical "bars", very short and cover a few frequency bars
     Vocalizations:
          - Cover a longer *horizontal* range, "distinct features"

    - OK: Audacity: Find program that plays the recordings -> narrow down where the vocalizations are
    - Spectrograms should be at maximum 2 seconds, initially, to look at how our models perform / inspect the
      data visually, that noise detection works
    - Other "noise": Some bird vocalizations are very short and cover a large spectrum, those are not not "vocalizations".
                    vocalizations have distinct features
    - OK: also add a constant when taking the log
     ** Sampling rate is 24kHz **


    #  What are S_trivial, S_clean? :
    #   S_trivial: Create a measure for silence -> automatically detect recordings where only one bird is vocalizing
    #   S_clean:   Same, we have to extract those parts

    The FFT part is based on:
        https://stackoverflow.com/questions/24382832/audio-spectrum-extraction-from-audio-file-by-python

'''

import soundfile as sf
#from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import copy
import re

from config import config
import utils



DAYS = ["2018-08-%d" % d for d in [14, 15, 16, 18, 19]]

def generate_recording_day_mapping():
    '''
        Look into all data folders and determine filenames and days for each recording number
    '''
    path = config["DATAPATH"]
    mapping = {}
    for day in DAYS:
        subpath = os.path.join(path, day)
        for thing in os.listdir(subpath):
            if os.path.isfile(os.path.join(subpath, thing)) and thing.endswith("SdrChannels.w64"):
                rec_nr = re.findall(r'b8p2male-b10o15female_(\d+)_SdrChannels.w64', thing)[0]
                rec_nr = int(rec_nr)
                assert not rec_nr in mapping.keys()
                mapping[rec_nr] = day
    return mapping

def channel_filename(recording_nr):
    path = config["DATAPATH"]
    day = RECORDING_DAYS[recording_nr]
    recording_base_str = "b8p2male-b10o15female_"+str(recording_nr)+"_"
    filenm = recording_base_str + "SdrChannels.w64"
    return os.path.join(path, day, filenm)

def strength_filename(recording_nr):
    path = config["DATAPATH"]
    day = RECORDING_DAYS[recording_nr]
    recording_base_str = "b8p2male-b10o15female_"+str(recording_nr)+"_"
    strength_filenm = recording_base_str + "SdrSignalStrength.w64"
    return os.path.join(path,day, strength_filenm)


RECORDING_DAYS = generate_recording_day_mapping()


class RecordingDataset():
    '''
    Class that manages access to the birdsong recordings
    Todo: Edit: Not sure we want to do that in this way. -- Generate shuffled batches for neural network training
    Needs to be done separately: Use two separate RecordingDatasets for train and validation data (and keep some recordings untouched, for testing data)
    Has a visualization function (plot_batch)
    '''


    def __init__(self, recordings:list=[3], window_size=512, overlap=0.875, max_freq=8000, min_freq=100, sequence_length=100, do_shuffle=True,
                 dB_signal_threshold_fraction=0.1, remove_noise=False, remove_simultaneous_vocalization=False):
        '''
        :param recordings: which recording files to read
        :param window_size: how many samples the fft window should span
        :param overlap between windows, in fractions of window size
        :param max_freq, min_freq: Band-pass filter the signal to this range.
                (It's already within the audible range, but the Nature paper reduces to between ? and 8kHz)
        :param sequence_length: Cut the recording into sequences of this length
        :param dB_signal_threshold_fraction: The norm of each spectrogram-sequence will be compared to this, and if the norm is below, the sequence is removed.
                                This is to filter out the parts of the recording without vocalizations. How to set this threshold is not quite clear though.
        :param remove_noise: If True, return S_clean (todo)
        :param remove_simultaneous_vocalization: If True, return S_trivial (todo)
        '''
        self.window_size = window_size
        self.overlap = overlap
        self.noverlap = int(np.floor(self.window_size * self.overlap))
        assert dB_signal_threshold_fraction > 0
        self.dB_signal_threshold_fraction = dB_signal_threshold_fraction
        assert  0 <= overlap < 1, "overlap as fraction of window size, so < 1 (0: no overlap)"
        valid_recording_nrs = sorted(list(RECORDING_DAYS.keys()))
        assert all([r in valid_recording_nrs for r in recordings]), "Only those recording numbers are available: "+str(valid_recording_nrs)
        self.recordings = recordings
        self.sequence_length = sequence_length
        self.do_shuffle = do_shuffle
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.remove_noise = remove_noise
        self.remove_simultaneous_vocalization = remove_simultaneous_vocalization


        self.samplerate = 24000 # 24 kHz

        # for batch loading:
        self._shuffled_recording_indices = None
        self._shuffled_recordings = np.array(recordings)
        self._cur_spectrogram_bird1 = None #  N x seq_length x nr_frequency_bins
        self._cur_spectrogram_bird2 = None #  N x seq_length x nr_frequency_bins
        self._cur_spectrogram_mic = None #  N x seq_length x nr_frequency_bins
        self._cur_audio_bird1 = None #  N x time points
        self._cur_audio_bird2 = None #  N x time points
        self._cur_audio_mic = None #  N x time points
        # self._cur_signal_strength = None # an N x seq_length array storing the average signal strength for each fourier window
        # self._cur_strength_raw = None


    def _read_recording(self, recording_nr):
        '''
        Reads a single recording and performs all the preprocessing
        The data can then be "shuffled" and returned, either as whole or in batches
        :param recording_nr
        :param remove_noise: Todo, cut or filter out noise -> S_clean
        :param remove_simultaneous_vocalization: Todo; if True return S_trivial

        :return:
            sets internal variables:
                - bird channel 1, spectrogram
                - bird channel 2, spectrogram
                - microphone channel, spectrogram
                - parallel signal strength array (reduced over the same window size and overlap)
        '''
        # 1. read the files
        filenm = channel_filename(recording_nr)
        strength_filenm = strength_filename(recording_nr)
        with open(filenm, 'rb') as f:
            Audiodata, _samplerate_bad = sf.read(f)
        self._cur_audio_mic = Audiodata[:, 0]
        self._cur_audio_bird1 = Audiodata[:, 1]
        self._cur_audio_bird2 = Audiodata[:, 2]
        del Audiodata

        with open(strength_filenm, 'rb') as f:
            self._cur_strength_raw, _str_samplerate = sf.read(f)

        # Filter to reduce low-frequency offset, and make fourier transform within our frequency range work better
        sos = signal.butter(10, [15, 8000], 'bp', fs=24000, output='sos')
        self._cur_audio_mic = signal.sosfilt(sos, self._cur_audio_mic)
        self._cur_audio_bird1 = signal.sosfilt(sos, self._cur_audio_bird1)
        self._cur_audio_bird2 = signal.sosfilt(sos, self._cur_audio_bird2)

        # 2. take the fourier transforms
        f, t, Sxx = signal.spectrogram(self._cur_audio_mic, self.samplerate, window=signal.hamming(self.window_size, sym=False), #window=signal.blackman(self.window_size),
                                       nfft=self.window_size, noverlap=self.noverlap, scaling="spectrum")
        self._cur_spectrogram_mic = {"frequencies": f, "t": t, "spectrogram": Sxx}
        f, t2, Sxx = signal.spectrogram(self._cur_audio_bird1, self.samplerate,window=signal.hamming(self.window_size, sym=False),# window=signal.blackman(self.window_size),
                                       nfft=self.window_size, noverlap=self.noverlap, scaling="spectrum")
        self._cur_spectrogram_bird1 = {"frequencies": f, "t": t2, "spectrogram": Sxx}
        f, t3, Sxx = signal.spectrogram(self._cur_audio_bird2, self.samplerate, window=signal.hamming(self.window_size, sym=False),#window=signal.blackman(self.window_size),
                                       nfft=self.window_size, noverlap=self.noverlap, scaling="spectrum")
        self._cur_spectrogram_bird2 = {"frequencies": f, "t": t3, "spectrogram": Sxx}
        assert np.all(t == t2)
        assert np.all(t == t3)
        # reduce to the frequency band we want
        if self.max_freq is not None or (self.min_freq is not None):
            if self.max_freq is not None:
                for spectro_dict in [self._cur_spectrogram_mic, self._cur_spectrogram_bird1, self._cur_spectrogram_bird2]:
                    good_ids = spectro_dict["frequencies"] <= self.max_freq
                    spectro_dict["spectrogram"] = spectro_dict["spectrogram"][good_ids]
                    spectro_dict["frequencies"] = spectro_dict["frequencies"][good_ids]
            if self.min_freq is not None:
                for spectro_dict in [self._cur_spectrogram_mic, self._cur_spectrogram_bird1, self._cur_spectrogram_bird2]:
                    good_ids = spectro_dict["frequencies"] >= self.min_freq
                    spectro_dict["spectrogram"] = spectro_dict["spectrogram"][good_ids]
                    spectro_dict["frequencies"] = spectro_dict["frequencies"][good_ids]

        # 3. signal strength has a different size of first dimension, but its first dimension corresponds to the time
        #       dimension of the other channels. --> stretch signalStrength array
        # self._cur_strength_raw = utils.interpolate1D(self._cur_strength_raw, (self._cur_spectrogram_mic["spectrogram"].shape[-1]))
        # self._cur_signal_strength = self._cur_strength_raw


        # 4a. placeholder for denoising
        #  Todo: @Others - edit: or is that solved now?
        # if self.return_clean: ...


        # 4b. placeholder for splitting into S_trivial vs. S_multiple
        #  Todo: @Others - edit: maybe do that elsewhere
        # if self.remove_simultaneous_vocalization: ...


        # 5. split all into sequence-length chunks (reshape)
        num_sequences = int(np.floor((self._cur_spectrogram_mic["spectrogram"].shape[1]) / self.sequence_length))
        for spectro_dict in [self._cur_spectrogram_mic, self._cur_spectrogram_bird1, self._cur_spectrogram_bird2]:
            num_freqs = spectro_dict["spectrogram"].shape[0]
            num_points =  spectro_dict["spectrogram"].shape[-1]
            assert num_freqs == len(spectro_dict["frequencies"])
            spectro_dict["spectrogram"] = spectro_dict["spectrogram"][..., : self.sequence_length * int(np.floor(num_points / self.sequence_length))]
            spectro_dict["spectrogram"] = spectro_dict["spectrogram"].transpose() # --> frequency in last dimension
            spectro_dict["spectrogram"] = spectro_dict["spectrogram"].reshape((num_sequences, self.sequence_length, num_freqs ))
            spectro_dict["t"] = spectro_dict["t"][ : self.sequence_length * int(np.floor(num_points / self.sequence_length))]
            spectro_dict["t"] = spectro_dict["t"].reshape(( num_sequences, self.sequence_length))
        # assert num_points == len(self._cur_signal_strength)
        # self._cur_signal_strength = self._cur_signal_strength[ :  self.sequence_length * int(np.floor(num_points / self.sequence_length)), :]
        # self._cur_signal_strength =  self._cur_signal_strength.reshape((num_sequences , self.sequence_length,  3))

        # 6. Remove recordings without vocalization:
        #   unit: dB? a type of log, that is for sure, because we have larger negative values

        mid_freq_ids = ((self._cur_spectrogram_mic["frequencies"] <= 6000) & (self._cur_spectrogram_mic["frequencies"] >= 2000))
        high_freq_ids = ((self._cur_spectrogram_mic["frequencies"] <= 8000) & (self._cur_spectrogram_mic["frequencies"] > 4000))
        main_freq_ids = ((self._cur_spectrogram_mic["frequencies"] <= 4000) & (self._cur_spectrogram_mic["frequencies"] >= 2000))
        average_power = np.linalg.norm(self._cur_spectrogram_mic["spectrogram"][:,:,mid_freq_ids], axis=(2))  # 0.001 #0.000001 # just from inspecting the histogram, 0.001 would be good - but recording 17 then doesnt have any signal?
        average_power_high = np.mean(np.abs(self._cur_spectrogram_mic["spectrogram"][:,:,high_freq_ids]), axis=(1,2))  # 0.001 #0.000001 # just from inspecting the histogram, 0.001 would be good - but recording 17 then doesnt have any signal?
        average_power_main = np.mean(np.abs(self._cur_spectrogram_mic["spectrogram"][:,:,main_freq_ids]), axis=(1,2))  # 0.001 #0.000001 # just from inspecting the histogram, 0.001 would be good - but recording 17 then doesnt have any signal?
        # max_average_power =  np.max(average_power, axis=1)
        average_power = np.linalg.norm(average_power, axis=(1))  # 0.001 #0.000001 # just from inspecting the histogram, 0.001 would be good - but recording 17 then doesnt have any signal?

        # * Determine the threshold as the dB-level above which a given fraction of all sequences lies
        #                                       /-- that log is crucial
        bins, bin_limits, sth_else = plt.hist(np.log(average_power), bins=150, log=True)
        total = np.sum( bins )
        thresh_amount = total * self.dB_signal_threshold_fraction
        thresh = bin_limits[0]
        sum_ = 0
        for b, lim in zip(bins[::-1], bin_limits[:-1][::-1]):
            sum_ +=  b
            if sum_ > thresh_amount:
                thresh = lim
                break

        ## Todo: 6e-6 is for: recordings=[11], window_size=512, overlap=0.1, max_freq=8000, min_freq=100,
        ##                  sequence_length=50,
        #  0.95e-7
        # good_ids = (average_power >= 1e-6) & (average_power_main >= average_power_high) #self.dB_signal_threshold_fraction
        good_ids = (average_power >= np.exp(thresh)) & (average_power_main >= average_power_high) #self.dB_signal_threshold_fraction
        assert np.sum(good_ids) < len(good_ids) * 0.5, "did not manage to filter out enough silence."
        # good_ids = average_power >=  self.dB_signal_threshold_fraction

        for spectro_dict in [self._cur_spectrogram_mic, self._cur_spectrogram_bird1, self._cur_spectrogram_bird2]:
            spectro_dict["spectrogram"] = spectro_dict["spectrogram"][good_ids, ...]
            spectro_dict["t"] = spectro_dict["t"][good_ids]
        # self._cur_signal_strength = self._cur_signal_strength[good_ids, ...]


        # 6. create shuffled or non-shuffled indices for this recording
        # -- very useless right now
        self._shuffled_recording_indices = np.array(list(np.arange((self._cur_spectrogram_mic["spectrogram"].shape[0]))))
        if self.do_shuffle:
            np.random.shuffle(self._shuffled_recording_indices)




    def yield_batches(self, batch_size=-1):
        ''' Todo: Not sure, maybe store to disk and do batch-loading elsewhere.
            Use via:
                >> for batch in bla.yield_batches():
                >>      ... # do sth with batch
            Returns:
                if batch_size < 0, return the whole recording at once.
                Shape of each of the returned spectrograms:
                    (batch_size, sequence_length, number_frequencies )
                Spectrograms are accessible at "spectrogram" in the returned dictionaries.
                Except the signal strength array, it's already an array and has shape (batch_size, sequence_length).

                Returns three dicts with spectrograms and one array of signal strengths, scaled to the same time points
        '''
        if self.do_shuffle:
            np.random.shuffle(self._shuffled_recordings)
        for rec in self._shuffled_recordings:
            self._read_recording(rec)
            if batch_size < 0:
                yield [self._cur_spectrogram_mic, self._cur_spectrogram_bird1, self._cur_spectrogram_bird2, rec,]
                       # self._cur_signal_strength]
            else:
                raise NotImplementedError("Batch-processing not implemented yet (todo; not too difficult though, just "
                                          "shuffle the indices of the whole recording, and create batch_size-sized batches from them)")


    def plot_batch(self, batch, base_path=""):
        ''' takes what's returned by yield_batches() in one step and creates spectrogram & strength plots'''
        # mic, bird1, bird2, signal_strength = batch
        mic, bird1, bird2, rec_nr = batch
        for spectrogram, name in [(mic, "mic"), (bird1, "bird1"), (bird2, "bird2")]:
            t = spectrogram["t"]
            f = spectrogram["frequencies"]
            Sxx = spectrogram["spectrogram"]
            for i, seq in enumerate(Sxx):
                t_ = t[i]
                plt.figure()
                plt.pcolormesh(t_, f, 10 * np.log10(2e-9 + seq.transpose()))  # dB spectrogram
                # plt.pcolormesh(t, f,Sxx) # Linear spectrogram
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.title(name + ', seq. '+str(i) + ' - Spectrogram with scipy.signal', size=16)

                # plt.pause(0.001)
                # plt.show()
                plt.savefig(base_path + "_" + str(i) + "_" + name )

    def save_batch(self, batch, base_path=""):
        ''' takes what's returned by yield_batches() in one step and stores each sequence to a separate .npy file.'''
        # mic, bird1, bird2, signal_strength = batch
        mic, bird1, bird2, recording_nr = batch
        path = os.path.join(base_path, "rec%02d/" % (recording_nr))
        utils.ensure_dir(path)
        for spectrogram, name in [(mic, "mic"), (bird1, "bird1"), (bird2, "bird2")]:
            t = spectrogram["t"]
            f = spectrogram["frequencies"]
            np.save(os.path.join(path, "frequencies.npy"), f)
            Sxx = spectrogram["spectrogram"]
            for i, seq in enumerate(Sxx):
                t_ = t[i]
                np.save(os.path.join(path,"seq_%04d_time.npy" % (i,)), t_)
                np.save(os.path.join(path,"seq_%04d_%s.npy" % (i, name)), Sxx)




def dont_run_just_annotation():
    path = config["DATAPATH"]
    filenames = ["2018-08-14/" + nm for nm in [
        "b8p2male-b10o15female_5_DAQmxChannels.w64",     # 1 channel   <--- Microphone! We don't use it, not aligned with the backpacks
        # SDR: One of the three channels is microphone
        "b8p2male-b10o15female_5_SdrSignalStrength.w64",  # <-- use to detect the radio noise.
                # Compare presentation:
                # 3 channels, values between -61 and 7
        "b8p2male-b10o15female_5_SdrCarrierFreq.w64",    # has 3 channels
        "b8p2male-b10o15female_5_SdrChannels.w64",       # <--- !! The microphone, aligned with the others.
            # 1st channel: microphone, 2nd: female backpack, 3rd: male backpack
            # 3 channels, not same length as DAQmx but similar (11 mio vs. 15 mio)
        "b8p2male-b10o15female_5_SdrReceiveFreq.w64",  # 3 channels. Has values between 301*10^6 and 307*10^6 --these are the min & max transmitter frequencies for the male bird (see .csv), whatever that means.
        "b8p2male-b10o15female_5_SdrChannelList.csv",
    ]]




def test_data_laoding():
    DS = RecordingDataset(recordings=[51], window_size=512, overlap=0.7, max_freq=8000, min_freq=100,
                          sequence_length=100, dB_signal_threshold_fraction=0.05 )
    for b in DS.yield_batches():
        DS.save_batch(b, base_path="../data/")
        DS.plot_batch(b, base_path="../plots/")
        mic, b1, b2, rec = b
        print("hello")



if __name__ == '__main__':
    test_data_laoding()