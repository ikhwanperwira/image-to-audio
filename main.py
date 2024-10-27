import cv2
import numpy as np
import pyaudio
import time
from collections import deque
from itertools import repeat, starmap

# Set the sample rate of the audio (in samples per second).
SAMPLE_RATE = 44100

# Number of frames per buffer to control audio data transfer frequency.
FRAMES_PER_BUFFER = 2048

# Set maximum buffer size to hold up to 2 seconds of audio data, lower buffer size means lower latency but prone to underflow.
MAX_BUFFER_SIZE = SAMPLE_RATE * 2

# Initialize a deque for audio data buffering with a max length of MAX_BUFFER_SIZE.
audio_buffer = deque(maxlen=MAX_BUFFER_SIZE)

# Create a buffer array to hold audio data fetched from the deque (initialization).
fetched_buffer = np.zeros(FRAMES_PER_BUFFER, dtype=np.float32)

# Overflow offset to adjust buffer size dynamically, flag to detect issues, and flag to start calculations.
overflow_offset = 0
underflow_bound = -999999
is_calculate_overflow = False

OVERFLOW_OFFSET_DECREMENT = 128
UNDERFLOW_BOUND_ADJUSTMENT = 1.01


def gen_sine_wave(freq, sample_rate=SAMPLE_RATE, frames_per_buffer=FRAMES_PER_BUFFER):
  """
  Generate a continuous sine wave for the specified frequency.
  :param freq: Frequency of the sine wave to generate
  :param sample_rate: The sample rate for audio playback
  :param frames_per_buffer: Number of frames to generate for each audio buffer
  """
  # Calculate the number of samples required to produce one full cycle of the sine wave.
  samples = int(sample_rate / freq)

  # Create one cycle of the sine wave based on frequency and sample rate.
  wave = np.sin(2 * np.pi * np.arange(samples) * freq / sample_rate)

  # Repeat the wave to fill at least one buffer's worth of data.
  while len(wave) < frames_per_buffer:
    # Double the length of wave each time to reach the required frames per buffer.
    wave = np.tile(wave, 2)

  return wave


def audio_callback(in_data, frame_count, time_info, status):
  """
  Callback function for pyaudio that outputs audio by fetching from audio_buffer.
  :param in_data: Input audio data (not used)
  :param frame_count: Number of frames to fetch from audio_buffer
  :param time_info: Timing information (not used)
  :param status: Audio stream status (not used)
  """
  global audio_buffer, fetched_buffer, overflow_offset, underflow_bound, is_calculate_overflow, UNDERFLOW_BOUND_ADJUSTMENT

  # Check if there are enough frames in the buffer to avoid underflow.
  if len(audio_buffer) < frame_count:
    print('Warning: Buffer underflow')  # Alert on underflow.

    # If audio calculation has started, adjust the underflow_bound to compensate for underflow.
    if is_calculate_overflow and overflow_offset != 0:
      underflow_bound = int(overflow_offset / UNDERFLOW_BOUND_ADJUSTMENT)

    # Reset overflow_offset since we just adjusted for it.
    overflow_offset = 0

    # Return the empty buffer to prevent playback disruption.
    return (fetched_buffer, pyaudio.paContinue)

  # Populate fetched_buffer with data from audio_buffer using popleft to maintain deque order.
  fetched_buffer = np.array(
      list(starmap(audio_buffer.popleft, repeat((), frame_count))), dtype=np.float32)

  # Return the filled buffer to play the audio.
  return (fetched_buffer, pyaudio.paContinue)


# Initialize webcam video capture.
cap = cv2.VideoCapture(0)

# Initialize pyaudio for audio stream.
p = pyaudio.PyAudio()

# Brief delay to allow everything to initialize properly.
time.sleep(1)

# Open an audio stream with specified settings.
audio_stream = p.open(format=pyaudio.paFloat32,
                      channels=1,
                      rate=SAMPLE_RATE,
                      output=True,
                      frames_per_buffer=FRAMES_PER_BUFFER // 2,
                      stream_callback=audio_callback)

# Start the audio stream.
audio_stream.start_stream()

# Additional delay to ensure the stream has time to start.
time.sleep(1)

# Begin processing frames and audio while the audio stream is active.
while audio_stream.is_active():
  # Capture a video frame.
  ret, frame = cap.read()
  if not ret:  # Exit if frame capture fails.
    break

  # Convert the frame to grayscale for intensity analysis.
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Calculate frequency based on average pixel intensity (scaled to a range of up to 1000 Hz).
  freq = np.mean(frame) / 255 * 1000

  # Check if audio_buffer is less than half full, accounting for the overflow_offset.
  if len(audio_buffer) < (MAX_BUFFER_SIZE // 2) + overflow_offset:
    # Generate a sine wave at the calculated frequency and add it to the buffer.
    next_wave = gen_sine_wave(freq, SAMPLE_RATE)
    audio_buffer.extend(next_wave)
  else:
    # If buffer is too full and overflow_offset reaches or exceeds underflow_bound, adjust for overflow.
    if overflow_offset >= underflow_bound:
      is_calculate_overflow = True  # Enable buffer adjustment calculations.
      # Decrease buffer size by OVERFLOW_OFFSET_DECREMENT to mitigate overflow.
      overflow_offset -= OVERFLOW_OFFSET_DECREMENT

  # Display the calculated frequency on the video frame.
  cv2.putText(frame, f'Frequency: {freq:.2f}', (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # Display the current overflow_offset on the video frame.
  cv2.putText(frame, f'Overflow offset: {overflow_offset}', (10, 60),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # Draw underflow bound
  cv2.putText(frame, f'Underflow bound: {underflow_bound}', (10, 90),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # Show the video frame with the frequency and offset annotations.
  cv2.imshow('frame', frame)

  # Check for the 'q' key to exit the loop and stop the program.
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release video capture resources.
cap.release()
# Close all OpenCV windows.
cv2.destroyAllWindows()
# Stop and close the audio stream.
audio_stream.stop_stream()
audio_stream.close()

# Terminate the pyaudio session.
p.terminate()
