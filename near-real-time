# pylint: disable = no-member
# pylint: disable = unused-argument
# pylint: disable = global-variable-not-assigned
# pylint: disable = missing-module-docstring
# pylint: disable = wrong-import-order
# pylint: disable = line-too-long
# pylint: disable = invalid-name
# pylint: disable = missing-function-docstring
# pylint: disable = global-statement

import pyaudio
import numpy as np
import time
from collections import deque
from itertools import starmap, repeat
import cv2

SAMPLE_RATE = 44100
MAX_BUFFER_LENGTH = int(SAMPLE_RATE*0.3)
FILL_RATIO_THRESHOLD = 0.5

last_produce_timestamp = time.time()
produce_interval = 1  # start from every 1 seconds


def generate_sine_wave_shape(freq, sample_rate=SAMPLE_RATE, min_chunk_length=2048):
  # Compute one-cycle length of wave shape
  num_samples = int(sample_rate / freq)

  # Generate wave shape with desired frequency
  t = np.linspace(0, 2 * np.pi, num_samples)
  wave_shape = np.sin(t)

  # Tile wave shape to match minimum chunk length
  num_tiles = int(min_chunk_length / num_samples)
  wave_shape = np.tile(wave_shape, num_tiles)

  return wave_shape


audio_buf = deque(maxlen=MAX_BUFFER_LENGTH)

fetched_buf = np.zeros(1024)


def audio_callback(in_data, frame_count, time_info, status):
  global fetched_buf, audio_buf, is_calculate_underflow, target_buffer_fill

  try:
    fetched_buf = np.array(
        list(starmap(audio_buf.popleft, repeat((), frame_count))), dtype=np.float32)
  except IndexError:  # underflow hit
    pass
    # print('Underflow frame count:', frame_count)

  return (fetched_buf, pyaudio.paContinue)


p = pyaudio.PyAudio()

time.sleep(3)

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                stream_callback=audio_callback,
                frames_per_buffer=1024)


cap = cv2.VideoCapture(0)

stream.start_stream()
while stream.is_active():
  # Get current state
  current_timestamp = time.time()
  fill_ratio = len(audio_buf) / MAX_BUFFER_LENGTH

  # Adjust target buffer fill based on current fill ratio to prevent underflow/overflow
  if fill_ratio > FILL_RATIO_THRESHOLD:
    # make it slower update
    produce_interval = 1
  elif fill_ratio < FILL_RATIO_THRESHOLD:
    # make it faster update
    produce_interval = 0

  ret, frame = cap.read()

  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  f = np.mean(frame) / 255 * 2000

  # Produce new audio data
  if current_timestamp - last_produce_timestamp > produce_interval:
    # f changing over time to simulate frequency change
    last_produce_timestamp = current_timestamp
    audio_buf.extend(generate_sine_wave_shape(f))

  # draw freq
  cv2.putText(frame, f'freq: {f:.2f}', (10, 50),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # draw fill ratio
  cv2.putText(frame, f'buff filled: {fill_ratio:.2f}', (10, 100),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

stream.stop_stream()
stream.close()

p.terminate()
