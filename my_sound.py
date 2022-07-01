# 音声を扱うための自作モジュール

import numpy as np
import matplotlib.pyplot as plt
import copy
import IPython.display
import numpy as np
import json


def sin_wave(
    freq: float = 440.,
    sec: float = 1.,
    sr: int = 44100,
    amp: float = 1.,
    fade_out: float = 0.
) -> np.ndarray:
    """
    sin波を生成する

    Args:
        freq (float): 周波数
        sec (float): 秒数
        sr (int): サンプリング周波数
        amp (float): 振幅
        fade_out (float): フェードアウト(秒数)

    Returns:
        np.ndarray: 生成したsin波
    """
    t = np.linspace(0, 2*np.pi * freq * sec, int(sr*sec))
    y = amp * np.sin(t)
    length = int(min((sr*sec, max((0, sr*fade_out)))))
    if length:
        y[-length:] *= np.linspace(1, 0, length)
    return y


def tick_insert(
    y: np.ndarray,
    locations: list,
    offset: int = 0,
    volume: float = 0.8,
    tick_sound: np.ndarray = sin_wave(3520, 0.03, 44100),
    plot: int = 0
) -> np.ndarray:
    """
    入力した音声データの入力した位置にtick音を挿入する

    Args:
        y (np.ndarray):
            tick音を挿入する音声データ
        locations (list):
            tick音を挿入する位置(index)
        offset (int):
            yとtick音のoffset
        volume (float):
            tick音の音量(yの最大値に対する割合)
        tick_sound (np.ndarray):
            挿入するtick音
        plot (int):
            何拍分プロットするか

    Returns:
        np.ndarray: tick音が挿入された音声データ
    """
    if max(0, plot):
        lim = np.abs(y).max()
        plt.figure(figsize=(12, 3))
        plt.plot(y[:locations[min(int(plot), len(locations))]], alpha=0.7)
        plt.vlines(
            locations[:min(int(plot), len(locations))],
            -lim, lim, color='r', alpha=0.7)

    tick_sound *= max(y)*volume
    length = len(tick_sound)
    y = copy.deepcopy(y)

    for loc in locations:
        loc += offset
        start = max((0, loc))
        end = min((loc + length, len(y)))
        if (start >= len(y)) or (end < 0):
            continue
        y[start:end] += tick_sound[:end - start]

    return y


def note2freq(note_number: int, A4: float = 440.) -> float:
    """
    midiノートナンバーを周波数に変換する.
    参考：A4 = 69

    Args:
        note_number (int): midiノートナンバー
        A4 (float, optional): 基準となるA4の周波数

    Returns:
        float: 周波数
    """
    return A4 * 2**((note_number - 69) / 12)

def Audio(audio: np.ndarray, sr: int):
    """
    以下から引用
    https://github.com/microsoft/vscode-jupyter/issues/1012#issuecomment-785410064

    vscodeのjupyterで音声を再生する.

    Use instead of IPython.display.Audio as a workaround for VS Code.
    `audio` is an array with shape (channels, samples) or just (samples,) for mono.
    """

    if np.ndim(audio) == 1:
        channels = [audio.tolist()]
    else:
        channels = audio.tolist()

    return IPython.display.HTML("""
        <script>
            if (!window.audioContext) {
                window.audioContext = new AudioContext();
                window.playAudio = function(audioChannels, sr) {
                    const buffer = audioContext.createBuffer(audioChannels.length, audioChannels[0].length, sr);
                    for (let [channel, data] of audioChannels.entries()) {
                        buffer.copyToChannel(Float32Array.from(data), channel);
                    }
            
                    const source = audioContext.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioContext.destination);
                    source.start();
                }
            }
        </script>
        <button onclick="playAudio(%s, %s)">Play</button>
    """ % (json.dumps(channels), sr))