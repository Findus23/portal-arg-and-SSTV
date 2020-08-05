from scipy.io import wavfile
from numpy import ndarray
from matplotlib import pyplot as plt

data: ndarray
samples_per_second, data = wavfile.read("sound/ambient/dinosaur3.wav")

samples_per_ms = samples_per_second // 1000
print(samples_per_ms)
plt.specgram(data[0 * samples_per_ms:20 * samples_per_second], Fs=samples_per_second, scale_by_freq=False)
plt.colorbar()
plt.show()
