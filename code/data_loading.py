'''
    Read and inspect the data provided by the supervisors (convert raw data to spectograms).
    Describe it in the methods sections. Describe your strategy of tackling the problem.
'''

import soundfile as sf
#from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os
#from scipy.io import wavfile  # scipy library to read wav files
import copy
import re

from config import config



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
    Todo: Can be used to generate shuffled batches for neural network training
    Needs to be done separately: Use two separate RecordingDatasets for train and validation data (and keep some recordings untouched, for testing data)
    Todo: Visualization function
    Todo: Do not read everything at once but allow parameterized reads and batches --> can train on the full data, otherwise not possible
    '''


    def __init__(self, recordings:list=[3], window_size=512, overlap=0.875, max_freq=8000, min_freq=350,  sequence_length=10, do_shuffle=True,
                 norm_threshold=1e-6):
        '''
        :param recordings: which recording files to read
        :param window_size: how many samples the fft window should span
        :param overlap between windows, in fractions of window size
        :param max_freq, min_freq: Todo. Band-pass filter the signal to this range. From the data: The transmitter frequency is limited.
        :param sequence_length: Cut the recording into sequences of this length
        :param norm_threshold: The norm of each spectrogram-sequence will be compared to this, and if the norm is below, the sequence is removed.
                                This is to filter out the parts of the recording without vocalizations. How to set this threshold is not quite clear though.
        '''
        self.window_size = window_size
        self.overlap = overlap
        self.noverlap = int(np.floor(self.window_size * self.overlap))
        self.norm_threshold = norm_threshold
        assert  0 <= overlap < 1, "overlap as fraction of window size, so < 1 (0: no overlap)"
        valid_recording_nrs = sorted(list(RECORDING_DAYS.keys()))
        assert all([r in valid_recording_nrs for r in recordings]), "Only those recording numbers are available: "+str(valid_recording_nrs)
        self.recordings = recordings
        self.sequence_length = sequence_length
        self.do_shuffle = do_shuffle
        self.max_freq = max_freq
        self.min_freq = min_freq
       # self.base_path =  config["DATAPATH"]

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
        self._cur_signal_strength = None # an N x seq_length array storing the average signal strength for each fourier window
        self._cur_strength_raw = None


    def _read_recording(self, recording_nr,  remove_noise=False, remove_simultaneous_vocalization=False):
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
        if len(Audiodata) > 20000000:
            Audiodata = Audiodata[:20000000]
        self._cur_audio_mic = Audiodata[:, 0]
        self._cur_audio_bird1 = Audiodata[:, 1]
        self._cur_audio_bird2 = Audiodata[:, 2]

        with open(strength_filenm, 'rb') as f:
            self._cur_strength_raw, _str_samplerate = sf.read(f)

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

        ## EDIT: signal strength has a different size of first dimension, so it can't correspond like that to the other channels.
        ## So since I don't know what it is at all, cannot split it into sequences
        # # 3. also average the signal strength to new size
        #     # a factor for calculations later:
        # _x = len(self._cur_audio_mic) / (self._cur_spectrogram_mic["spectrogram"].shape[1]) #  a float
        # self._cur_signal_strength = np.zeros((self._cur_spectrogram_mic["spectrogram"].shape[1]), float)
        # for i in range((self._cur_spectrogram_mic["spectrogram"].shape[1])):
        #     lower = int(np.floor(i * (_x)))
        #     higher = int(np.floor((i + 1) * (_x)))
        #     self._cur_signal_strength[i] = np.mean(np.array(self._cur_strength_raw[ lower : higher]))
        #     # ... don't know how to do this with little effort and without knowing exactly how windowing works in the spectrogram function above.

        # 4a. placeholder for denoising
        #  Todo: @Others

        # 4b. placeholder for splitting into S_trivial vs. S_multiple
        #  Todo: @Others

        # 5. split all into sequence-length chunks (reshape)
        num_sequences = int(np.floor((self._cur_spectrogram_mic["spectrogram"].shape[1]) / self.sequence_length))
        for spectro_dict in [self._cur_spectrogram_mic, self._cur_spectrogram_bird1, self._cur_spectrogram_bird2]:
            num_freqs = spectro_dict["spectrogram"].shape[0]
            num_points =  spectro_dict["spectrogram"].shape[-1]
            assert num_freqs == len(spectro_dict["frequencies"])
            spectro_dict["spectrogram"] = spectro_dict["spectrogram"][..., : num_sequences * int(np.floor(num_points / num_sequences))]
            spectro_dict["spectrogram"] = spectro_dict["spectrogram"].transpose() # --> frequency in last dimension
            spectro_dict["spectrogram"] = spectro_dict["spectrogram"].reshape((num_sequences, self.sequence_length, num_freqs ))
            spectro_dict["t"] = spectro_dict["t"][ : num_sequences * int(np.floor(num_points / num_sequences))]
            spectro_dict["t"] = spectro_dict["t"].reshape(( num_sequences, self.sequence_length))
        # assert num_points == len(self._cur_signal_strength)
        # self._cur_signal_strength = self._cur_signal_strength[..., :  num_sequences * int(np.floor(num_points / num_sequences))]
        # self._cur_signal_strength =  self._cur_signal_strength.reshape((-1, num_sequences ))

        # # for debugging, remove later:
        # for sidx in range(10):# range(spectro_dict["spectrogram"].shape[2]):
        #     s = spectro_dict["spectrogram"][:, :, sidx]
        #     plt.figure()
        #     plt.plot(range(len(s)), np.linalg.norm(s, axis=1))
        #     plt.pause(0.001)
        #     plt.show()


        # 6. Remove recordings without vocalization:
        average_power = np.linalg.norm(self._cur_spectrogram_mic["spectrogram"], axis=(1,2))
        # plt.hist(average_power, bins=20, log=True)
        good_ids = average_power > self.norm_threshold # 0.001 #0.000001 # just from inspecting the histogram, 0.001 would be good - but recording 17 then doesnt have any signal?
        for spectro_dict in [self._cur_spectrogram_mic, self._cur_spectrogram_bird1, self._cur_spectrogram_bird2]:
            spectro_dict["spectrogram"] = spectro_dict["spectrogram"][good_ids, ...]
            spectro_dict["t"] = spectro_dict["t"][good_ids]

        for idx, seq in enumerate(self._cur_spectrogram_mic["spectrogram"]):
            pass

        # 6. create shuffled or non-shuffled indices for this recording
        self._shuffled_recording_indices = np.array(list(np.arange((self._cur_spectrogram_mic["spectrogram"].shape[0]))))
        if self.do_shuffle:
            np.random.shuffle(self._shuffled_recording_indices)




    def yield_batches(self, batch_size=-1):
        ''' Use via:
                >> for batch in bla.yield_batches():
                >>      ... # do sth with batch
            Returns:
                if batch_size < 0, return the whole recording at once.
                Shape of each of the returned spectrograms:
                    (batch_size, sequence_length, number_frequencies )
                Spectrograms are accessible at "spectrogram" in the returned dictionaries.
                Except the signal strength array, it's already an array and has shape (batch_size, sequence_length).
        '''
        if self.do_shuffle:
            np.random.shuffle(self._shuffled_recordings)
        for rec in self._shuffled_recordings:
            self._read_recording(rec)
            if batch_size < 0:
                yield [self._cur_spectrogram_mic, self._cur_spectrogram_bird1, self._cur_spectrogram_bird2]
                     #  self._cur_signal_strength]
            else:
                raise NotImplementedError("Batch-processing not implemented yet (todo; not too difficult though, just "
                                          "shuffle the indices of the whole recording, and create batch_size-sized batches from them)")


    def plot_batch(self, batch, base_path=""):
        ''' takes what's returned by yield_batches() in one step and creates spectrogram & strength plots'''
        mic, bird1, bird2 = batch
        for spectrogram, name in [(mic, "mic"), (bird1, "bird1"), (bird2, "bird2")]:
            t = spectrogram["t"]
            f = spectrogram["frequencies"]
            Sxx = spectrogram["spectrogram"]
            for i, seq in enumerate(Sxx):
                t_ = t[i]
                plt.figure()
                plt.pcolormesh(t_, f, 10 * np.log10(1. + seq.transpose()))  # dB spectrogram
                # plt.pcolormesh(t, f,Sxx) # Linear spectrogram
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.title(name + ', seq. '+str(i) + ' - Spectrogram with scipy.signal', size=16)

                plt.pause(0.001)
                #plt.show()
                plt.savefig(base_path + "_" + str(i) + "_" + name )

                # # for debugging: Why is the density only in the last dimension?
                # plt.figure()
                # plt.hist(np.linalg.norm(seq, axis=1))
                # plt.pause(0.001)
                # plt.show()

        # for i, seq in enumerate(sig_strength):
        #         plt.figure()
        #         plt.ylabel('Signal strength')
        #         plt.xlabel('Time [sec]')
        #         plt.title('Signal strenght, seq. '+str(i) , size=16)
        #         plt.show()
        #         plt.savefig(base_path + "_" + str(i) + "_" + "strength" )



def test_data_laoding():
    DS = RecordingDataset(recordings=[3], window_size=512, overlap=0.875, max_freq=8000, min_freq=350,  sequence_length=10, do_shuffle=True)
    for b in DS.yield_batches():
        DS.plot_batch(b)
        mic, b1, b2 = b
        print("hello")

     # todo


def first_try_plot():
    path = config["DATAPATH"]
    #  There is an online documentation: Link is in the presentation (try to get the newest version).
    #  https://www.authorea.com/users/53192/articles/321045-sdr-birdrec-software-documentation
    #

    # - Todo: Vocalizations: betw. 50 and 200 ms --> check that you reduce the FFT window
    # - Todo: Find program that plays the recordings -> narrow down where the vocalizations are
    # - Spectrograms should be at maximum 2 seconds, initially, to look at how our models perform / inspect the
    #   data visually, that noise detection works
    # - check for a small online GUI to inspect this
    # - Other "noise": Some bird vocalizations are very short and cover a large spectrum, those are not not "vocalizations".
    #                 vocalizations have distinct features
    # - Todo: also add a constant when taking the log
    #  ** Sampling rate is 24kHz **

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

    #
    #  What are S_trivial, S_clean? :
    #   S_trivial: Create a measure for silence -> automatically detect recordings where only one bird is vocalizing
    #   S_clean:   Same, we have to extract those parts
    #

    for fn in filenames[:-1]:
        filename0 = os.path.join(path, fn)
        with open(filename0, 'rb') as f:
                Audiodata, samplerate = sf.read(f)

        ########################

        # **  From:
        #       https://stackoverflow.com/questions/24382832/audio-spectrum-extraction-from-audio-file-by-python

        #fs, Audiodata = wavfile.read(filename0)

        # Plot the audio signal in time
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(Audiodata[:10000000])
        plt.title('Audio signal in time', size=16)
        # # spectrum
        # from scipy.fftpack import fft  # fourier transform
        # n = len(Audiodata)
        # AudioFreq = fft(Audiodata, axis=0)
        # AudioFreq = AudioFreq[0:int(np.ceil((n + 1) / 2.0))]  # Half of the spectrum
        # MagFreq = np.abs(AudioFreq)  # Magnitude
        # MagFreq = MagFreq / float(n)
        # # power spectrum
        # MagFreq = MagFreq ** 2
        # if n % 2 > 0:  # ffte odd
        #     MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
        # else:  # fft even
        #     MagFreq[1:len(MagFreq) - 1] = MagFreq[1:len(MagFreq) - 1] * 2
        #
        # plt.figure()
        # freqAxis = np.arange(0, int(np.ceil((n + 1) / 2.0)), 1.0) * (samplerate / n);
        # plt.plot(freqAxis / 1000.0, 10 * np.log10(MagFreq))  # Power spectrum
        # plt.xlabel('Frequency (kHz)');
        # plt.ylabel('Power spectrum (dB)');
        #
        # # Spectrogram
        # from scipy import signal
        # N = 512  # Number of point in the fft
        # if len(Audiodata.shape) == 1:
        #     Audiodata = Audiodata[:, np.newaxis]
        # for i in range(Audiodata.shape[1]):
        #         f, t, Sxx = signal.spectrogram(Audiodata[:10000000, i], samplerate, window=signal.blackman(N), nfft=N)
        #         plt.figure()
        #         plt.pcolormesh(t, f, 10 * np.log10(Sxx))  # dB spectrogram
        #         # plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
        #         plt.ylabel('Frequency [Hz]')
        #         plt.xlabel('Time [seg]')
        #         plt.title('Spectrogram with scipy.signal', size=16);

        plt.pause(0.001)

    print("breakpoint")


if __name__ == '__main__':
    test_data_laoding()
    #first_try_plot()