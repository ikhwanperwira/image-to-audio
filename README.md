# Real-Time Audio Generation from Webcam Intensity

This project captures video frames from a webcam, processes the images to extract average grayscale intensity, and converts that intensity into a continuous audio sine wave. The audio frequency is determined by the intensity of light detected by the camera, creating a unique sound based on the visual environment.

## Project Overview

The key innovation in this project lies in the dynamic management of audio buffer parameters to achieve near-real-time audio playback. The parameters `overflow_offset` and `underflow_bound` work together to ensure a smooth audio experience, adapting to variations in frame processing speed and audio playback demands.

![thumbnail](./thumbnail.mkv)

### Key Concepts:

- **Overflow Offset**: A dynamic parameter that adjusts the audio buffer size. Lower values mean faster processing but may lead to buffer underflow, resulting in audio interruptions.
  
- **Underflow Bound**: The threshold for adjusting the overflow offset. If underflow occurs, the underflow bound increases, allowing for a larger buffer size and smoother playback.

### Audio Generation:
The project uses a simple aggregation technique by averaging the pixel intensities of the entire grayscale image to produce an amplitude. This amplitude is then mapped to a frequency for a sine wave that is played as audio.

## Features

- Real-time audio generation based on webcam light intensity.
- Adjustable parameters for optimizing audio buffer management.
- Simple averaging technique to convert pixel intensity to frequency.
- Visual feedback via OpenCV displaying current frequency and buffer status.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- PyAudio

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/repo-name.git
   cd repo-name
   ```

2. Install the required packages:

   ```bash
   pip install opencv-python numpy pyaudio
   ```

## Usage

1. Ensure your webcam is connected.
2. Run the script:

   ```bash
   python audio_visualizer.py
   ```

3. The program will start capturing video from the webcam. It will calculate the average intensity of the grayscale frame and convert it into a sine wave audio output.
4. Press 'q' to exit the program.

## Algorithm Explanation

The algorithm employs a feedback mechanism between `overflow_offset` and `underflow_bound`:

- **Buffer Underflow Handling**:
  - If underflow occurs, the `underflow_bound` is adjusted to be 10% greater than the last value of `overflow_offset`. This effectively increases the threshold for the overflow offset to decrease, providing a buffer zone for audio playback.

- **Buffer Overflow Handling**:
  - As audio data is fed into the buffer, if the buffer is approaching its maximum capacity, the `overflow_offset` is decremented by a defined constant (e.g., 128). This adjustment aims to reduce potential delays in audio playback.

- **Dynamic Adjustment**: 
  - Over time, the `overflow_offset` finds an optimal value that minimizes underflow occurrences without impacting real-time performance. The algorithm operates independently of frame-per-second (FPS) calculations.

## Visual Feedback

During execution, the program displays:

- The current calculated frequency derived from the grayscale pixel intensity.
- The current `overflow_offset`, indicating how much it has been adjusted.
- The `underflow_bound`, showcasing the threshold for buffer adjustment.

This visual feedback aids in understanding how light intensity impacts sound output in real-time.

## Conclusion

This project demonstrates an innovative approach to integrating audio synthesis with visual inputs, showcasing the versatility of real-time processing in Python. By fine-tuning buffer management parameters, it achieves a seamless audio experience based on the visual light intensity captured by the webcam.

For further enhancements, consider adding support for multiple audio channels, frequency modulation, or visual representations of the audio output.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

### Instructions to Use the README
1. Replace `yourusername` and `repo-name` in the clone command with your actual GitHub username and repository name.
2. Adjust any details regarding installation, features, or usage based on your specific project setup.
3. Add any additional sections or information that you think might be helpful for users or contributors.

Feel free to modify any part of the README to better fit your project's style or needs!