import tdt
import numpy as np
import pandas as pd
import datetime
import scipy.signal
from photometry_functions import *

#folder = "/home/matias/Experiments/pilots/photometry/MLA074-220414-141939/"

def get_tdt_data(folder, decimate=True, decimate_factor = 10, remove_start=False, verbose=False):
  '''
  get_tdt_data is a function to retrieve the data streams as saved by TDT system
  it uses tdt package and will retrieve the complete duration
  returns a data frame with UTC timestamp, time in seconds, and signal values for each channel
  '''
  if verbose:
    print(f"Reading data from {folder}")
  data = tdt.read_block(folder)
  total_samples = len(data.streams._405A.data)
  fs = data.streams._405A.fs
  # inverse sampling frequency in Hz
  total_seconds = len(data.streams._405A.data)/fs
  start_date = data.info.start_date
  end_date = data.info.stop_date
  sampling_interval = 1 / fs
  
  _405A_data = data.streams._405A.data
  _465A_data = data.streams._465A.data
  
  if decimate:
    sampling_interval = sampling_interval * decimate_factor
    total_samples = np.ceil(total_samples / decimate_factor)
    # Decimate
    _405A_data = scipy.signal.decimate(_405A_data, decimate_factor, ftype="fir")
    _465A_data = scipy.signal.decimate(_465A_data, decimate_factor, ftype="fir")
  # UTC datetime
  datetime = pd.date_range(start_date, end_date, periods=total_samples)
  # time_delta = datetime - start_date
  # time_delta = time_delta / np.timedelta64(1, 's')
  # using np works for a seconds range
  time_np = np.arange(0, total_seconds, sampling_interval)

  df = pd.DataFrame({
    "utc_datetime" : datetime,
    "time_seconds" : time_np,
    "_405" : _405A_data,
    "_465" : _465A_data
  })
  
  # TODO: improve the check for third channel
  
  if remove_start:
    # this will have the times when each laser was turned on
    laser_on_times = data.scalars.Fi1i.ts
    # remove from the max moment when leds are on plus 5 seconds
    remove_before = np.ceil(max(laser_on_times) * fs) + 5 * fs
    # we only care about the max here 
    # because we end up removing everything before this
    df = df.iloc[:remove_before]
  
  return df


def get_cam_timestamps(folder, cam_name="Cam1", verbose=False):
  '''
  get_cam_timestamps is a function to retrieve timestamps from a camera 
  using the data streams as saved by TDT system.
  it uses tdt package and will retrieve the complete duration
  cam_name: string with the camera name as saved configured in Synapse software
  returns the timestamp onset
  '''
  if verbose:
    print(f"Reading data from {folder}")
  data = tdt.read_block(folder)
  return data.epocs[cam_name].onset

def calculate_zdFF(photo_data, n_remove=5000):
  photo_subset = photo_data.loc[n_remove:].copy()
  # try to estimate the sampling rate
  one_second = int(1 / photo_subset["time_seconds"].diff().values[-1])
  # we might need to fix the issues here with size errors
  photo_subset["zdFF"] = get_zdFF(
    photo_subset._405, 
    photo_subset._465, 
    smooth_win=one_second, 
    remove=1)
  
  final_data =  pd.merge(photo_data, photo_subset["zdFF"], 
                         how="left", 
                         left_index=True, right_index=True)
  # make them zero
  final_data["zdFF"].fillna(0, inplace=True)
  return final_data
