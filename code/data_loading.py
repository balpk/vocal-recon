'''
    Read and inspect the data provided by the supervisors (convert raw data to spectograms).
    Describe it in the methods sections. Describe your strategy of tackling the problem.
'''

import soundfile as sf
import os
from scipy.io import wavfile  # scipy library to read wav files
import numpy as np

from config import config

def load_recording(day=14, recording=3, more_parameters_that_are_useful=None):
    # todo
    day = "2018-08-"+str(day)
    recording_base_str = "b8p2male-b10o15female_"+str(recording)+"_"
    filenm = recording_base_str + "SdrChannels.w64"
    strength_filenm = recording_base_str + "SdrSignalStrength.w64"



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
        # spectrum
        from scipy.fftpack import fft  # fourier transform
        n = len(Audiodata)
        AudioFreq = fft(Audiodata, axis=0)
        AudioFreq = AudioFreq[0:int(np.ceil((n + 1) / 2.0))]  # Half of the spectrum
        MagFreq = np.abs(AudioFreq)  # Magnitude
        MagFreq = MagFreq / float(n)
        # power spectrum
        MagFreq = MagFreq ** 2
        if n % 2 > 0:  # ffte odd
            MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
        else:  # fft even
            MagFreq[1:len(MagFreq) - 1] = MagFreq[1:len(MagFreq) - 1] * 2

        plt.figure()
        freqAxis = np.arange(0, int(np.ceil((n + 1) / 2.0)), 1.0) * (samplerate / n);
        plt.plot(freqAxis / 1000.0, 10 * np.log10(MagFreq))  # Power spectrum
        plt.xlabel('Frequency (kHz)');
        plt.ylabel('Power spectrum (dB)');

        # Spectrogram
        from scipy import signal
        N = 512  # Number of point in the fft
        if len(Audiodata.shape) == 1:
            Audiodata = Audiodata[:, np.newaxis]
        for i in range(Audiodata.shape[1]):
                f, t, Sxx = signal.spectrogram(Audiodata[:10000000, i], samplerate, window=signal.blackman(N), nfft=N)
                plt.figure()
                plt.pcolormesh(t, f, 10 * np.log10(Sxx))  # dB spectrogram
                # plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [seg]')
                plt.title('Spectrogram with scipy.signal', size=16);

        plt.show()

    print("breakpoint")


if __name__ == '__main__':
    first_try_plot()