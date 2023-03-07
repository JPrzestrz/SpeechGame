# Libraries needed
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# vars 
DATASET_PATH = 'data/mini_speech_commands'
data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
AUTOTUNE = tf.data.AUTOTUNE

# Function that transformed the raw WAV audio files of the data set into audio tensors
def decode_audio(audio_binary):
      '''
      Function for transforming raw WAV audio into audio tf tensors 
      '''
      # Decode WAV-encoded audio files to `float32` tensors, normalized
      # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
      audio, _ = tf.audio.decode_wav(contents=audio_binary)
      # Since all the data is single channel (mono), drop the `channels`
      # axis from the array.
      return tf.squeeze(audio, axis=-1)

def get_label(file_path):
      '''
      Split file paths into tf.RaggedTensor s
      '''
      parts = tf.strings.split(input=file_path,sep=os.path.sep)
      # Note: You'll use indexing here instead of tuple unpacking to enable this
      # to work in a TensorFlow graph.
      return parts[-2]

def get_waveform_and_label(file_path):
      '''
      The input data is the name of the WAV audio file.
      The output is a tuple of audio and label tensors ready for supervised learning.
      '''
      label = get_label(file_path)
      audio_binary = tf.io.read_file(file_path)
      waveform = decode_audio(audio_binary)
      return waveform, label

def get_spectrogram(waveform):
      '''
      Function for changing waves into spectograms 
      '''
      # Zero-padding for an audio waveform with less than 16,000 samples.
      input_len = 16000
      waveform = waveform[:input_len]
      zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
      # Cast the waveform tensors' dtype to float32.
      waveform = tf.cast(waveform, dtype=tf.float32)
      # Concatenate the waveform with `zero_padding`, which ensures all audio
      # clips are of the same length.
      equal_length = tf.concat([waveform, zero_padding], 0)
      # Convert the waveform to a spectrogram via a STFT.
      spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
      # Obtain the magnitude of the STFT.
      spectrogram = tf.abs(spectrogram)
      # Add a `channels` dimension, so that the spectrogram can be used
      # as image-like input data with convolution layers (which expect
      # shape (`batch_size`, `height`, `width`, `channels`).
      spectrogram = spectrogram[..., tf.newaxis]
      return spectrogram

def plot_spectrogram(spectrogram, ax):
      '''
      Function for ploting the spectograms 
      '''
      if len(spectrogram.shape) > 2:
            assert len(spectrogram.shape) == 3
            spectrogram = np.squeeze(spectrogram, axis=-1)
      # Convert the frequencies to log scale and transpose, so that the time is
      # represented on the x-axis (columns).
      # Add an epsilon to avoid taking a log of zero.
      log_spec = np.log(spectrogram.T + np.finfo(float).eps)
      height = log_spec.shape[0]
      width = log_spec.shape[1]
      X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
      Y = range(height)
      ax.pcolormesh(X, Y, log_spec)

# Function that transforms the waveform data set into spectrograms 
# and their corresponding labels as integer identifiers
def get_spectrogram_and_label_id(audio, label):
      '''
      Tranforming waveform into spectograms and labels 
      '''
      spectrogram = get_spectrogram(audio)
      label_id = tf.argmax(label == commands)
      return spectrogram, label_id

# Repeating the training kit preprocessing 
# on the validation and test kits 
def preprocess_dataset(files):
      files_ds = tf.data.Dataset.from_tensor_slices(files)
      output_ds = files_ds.map(
            map_func=get_waveform_and_label,
            num_parallel_calls=AUTOTUNE)
      output_ds = output_ds.map(
            map_func=get_spectrogram_and_label_id,
            num_parallel_calls=AUTOTUNE)
      return output_ds
