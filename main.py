from typing import List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.optimize import curve_fit


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def peak_finder(signal: ndarray, samples_per_second: int, plot: bool = False, cut_borders: bool = False) -> float:
    if cut_borders:
        signal = signal[20:]
        signal = signal[:-20]

    sp: ndarray = np.fft.rfft(signal)
    real_spectrum: ndarray = np.abs(sp)
    freq = np.fft.rfftfreq(len(signal), 1 / samples_per_second)
    peak = freq[np.argmax(real_spectrum)]
    # return peak
    try:
        popt, pcov = curve_fit(gaus, freq, real_spectrum, p0=[3.5e5, peak, 1.6e2], maxfev=2000)
    except RuntimeError:
        return 0
    # print(popt[1], peak)
    if plot:
        newx = np.linspace(0, np.max(freq), 1000)
        plt.figure()
        plt.plot(freq, real_spectrum, linewidth=0.3)
        plt.plot(newx, gaus(newx, *popt), linewidth=0.3)
        plt.show()

    return float(popt[1])


def even_parity_check(data: List[bool], parity: bool) -> bool:
    total = 0
    for bit in data:
        if bit:
            total += 1
    print(total)
    print(total % 2)
    even = total % 2 == 0
    parity_bit = not even
    return parity_bit == parity


assert even_parity_check([False, False, True, True, False, True, False], True)


def lsb_first_binary(data: List[bool]) -> int:
    i = 1
    total = 0
    for bit in data:
        if bit:
            total += i
        i *= 2
    return total


# 4 + 8 + 32 = 44
assert lsb_first_binary([False, False, True, True, False, True, False]) == 44


def main():
    data: ndarray
    sunset = False
    samples_per_second, data = wavfile.read("sound/ambient/dinosaur3.wav")
    # samples_per_second, data = wavfile.read("SSTV_sunset_audio.wav")

    spm = samples_per_second // 1000

    # silence_len = 134 if sunset else 510  # ms
    silence_len = 885 if sunset else 598  # ms
    header_len = 300  # ms
    # header_break_len = 10  # ms
    header_break_len = 30 if sunset else 10  # ms
    vis_bit_length = 30  # ms

    header1_start = silence_len
    header1_end = header1_start + header_len
    header_break_start = header1_end
    header_break_end = header_break_start + header_break_len
    header2_start = header_break_end
    header2_end = header2_start + header_len

    headers = [
        data[header1_start * spm: header1_end * spm],
        data[header2_start * spm: header2_end * spm]
    ]

    for header in headers:
        header_freq = peak_finder(header, samples_per_second)
        print(header_freq)
        assert 1895 < header_freq < 1905  # should be 1900 hz

    print(header_break_start)
    header_break = data[header_break_start * spm: header_break_end * spm]
    header_freq = peak_finder(header_break, samples_per_second)
    print(header_freq)
    assert 1195 < header_freq < 1206  # should be 1200 hz

    print("--- VIS start ---")
    pos = header2_end
    bits: List[bool] = []
    for i in range(10):
        print(pos, pos + vis_bit_length)
        bit_data = data[pos * spm: (pos + vis_bit_length) * spm]
        pos += vis_bit_length
        pixel_freq = peak_finder(bit_data, samples_per_second, cut_borders=True)
        print(pixel_freq)
        if i in [0, 9]:  # start bit at 1200 Hz, stop bit at 1200 Hz
            assert 1190 < pixel_freq < 1210
            continue
        if 1090 < pixel_freq < 1110:
            bits.append(True)
        elif 1290 < pixel_freq < 1310:
            bits.append(False)
        else:
            raise ValueError(f"{pixel_freq} hz is not an unique bit")
    print("---")
    print(pos)
    print(bits)
    print(bits[:7], bits[7])
    assert even_parity_check(bits[:7], bits[7])
    mode = lsb_first_binary(bits[:7])
    print(mode)
    if mode == 44:
        martin_m1 = True
        robot36 = False
        row_width = 447
        print("Martin M1 detected")
    elif mode == 8:
        martin_m1 = False
        robot36 = True
        row_width = 150
        print("Robot 36 detected")
    else:
        martin_m1 = False
        robot36 = False
        row_width = 0
        print(f"unknown mode: {mode}")
        exit()
    row_width_padded = row_width + 100

    image = []
    overlap = 40
    row = []
    while True:
        pixel_data = data[pos * spm - overlap: (pos + 1) * spm + overlap]
        pixel_data = pixel_data * np.hamming(len(pixel_data))
        pos += 1
        if len(pixel_data) != 1 * spm + overlap * 2:
            break
        pixel_freq = peak_finder(pixel_data, samples_per_second, False)
        if pixel_freq < 1300:
            if len(row):
                if len(row) < row_width_padded:
                    print(row_width_padded - len(row))
                    row.extend([0] * (row_width_padded - len(row)))
                if len(row) > row_width_padded:
                    row = row[:250]
                image.append(row)
                print(len(row), pos)
            row = []
        else:
            row.append(pixel_freq)

    image = np.asarray(image)
    print(image.shape)
    if martin_m1:
        grid = 147
        image = np.array([image[::, (2 * grid):(3 * grid)], image[::, 0:grid], image[::, grid:2 * grid]])
        image -= 1500
        image /= 2300 - 1500
        image = np.moveaxis(image, 0, -1)
        image = image.clip(0, 1)
    plt.imshow(image, aspect="auto")
    # plt.imsave("output.png", image)
    # plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
