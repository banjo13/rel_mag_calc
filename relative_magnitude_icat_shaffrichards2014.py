# -------------------------
# Imports
# -------------------------

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

import obspy
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.event import Catalog
from obspy.signal.cross_correlation import xcorr_pick_correction, correlate

# Initialize FDSN client
client = Client("IRIS")

# -------------------------
# Helper Functions
# -------------------------

def preprocess_stream(st, freqmin=2.0, freqmax=10.0):
    """Preprocess a Stream by detrending, demeaning, and bandpass filtering."""
    st.detrend("demean")
    st.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
    return st

def _rms(array):
    """Compute root-mean-square of a NumPy array."""
    return np.sqrt(np.mean(np.square(array)))

def _get_pick_for_station(event, station, use_s_picks):
    """Return first pick for a given station."""
    picks = [p for p in event.picks if p.waveform_id.station_code == station]
    if not picks:
        return None
    picks.sort(key=lambda p: p.time)
    for pick in picks:
        if pick.phase_hint and pick.phase_hint[0].upper() == 'S' and not use_s_picks:
            continue
        return pick
    return None

def _get_signal_and_noise(stream, event, seed_id, noise_window, signal_window, use_s_picks):
    """Extract signal and noise amplitudes for a given trace and event."""
    station = seed_id.split('.')[1]
    pick = _get_pick_for_station(event, station, use_s_picks)
    if pick is None:
        return None, None, None

    tr = stream.select(id=seed_id).merge()
    if len(tr) == 0:
        return None, None, None
    tr = tr[0]

    noise = tr.slice(starttime=pick.time + noise_window[0],
                     endtime=pick.time + noise_window[1]).data
    noise_amp = _rms(noise)

    if np.isnan(noise_amp):
        noise_amp = None

    signal = tr.slice(starttime=pick.time + signal_window[0],
                      endtime=pick.time + signal_window[1]).data
    if len(signal) == 0:
        return noise_amp, None, None

    signal_amp = _rms(signal)
    return noise_amp, signal_amp, signal.std()

def relative_amplitude(st1, st2, event1, event2, noise_window=(-20, -1),
                       signal_window=(-0.5, 20), min_snr=1.5, use_s_picks=False):
    """Compute relative amplitudes between two streams."""
    seed_ids = {tr.id for tr in st1}.intersection({tr.id for tr in st2})
    amplitudes = {}
    snrs_1 = {}
    snrs_2 = {}

    for seed_id in seed_ids:
        noise1, signal1, std1 = _get_signal_and_noise(
            st1, event1, seed_id, noise_window, signal_window, use_s_picks)
        noise2, signal2, std2 = _get_signal_and_noise(
            st2, event2, seed_id, noise_window, signal_window, use_s_picks)

        noise1 = noise1 or noise2
        noise2 = noise2 or noise1

        if noise1 is None or noise2 is None:
            continue
        if signal1 is None or signal2 is None:
            continue

        snr1 = np.nan_to_num(signal1 / noise1)
        snr2 = np.nan_to_num(signal2 / noise2)

        if snr1 < min_snr or snr2 < min_snr:
            continue

        ratio = std2 / std1
        amplitudes[seed_id] = ratio
        snrs_1[seed_id] = snr1
        snrs_2[seed_id] = snr2

    return amplitudes, snrs_1, snrs_2

def relative_magnitude(st1, st2, event1, event2, noise_window=(-20, -1),
                       signal_window=(-0.5, 20), min_snr=5.0, min_cc=0.7,
                       use_s_picks=False, correlations=None, shift=2.0,
                       return_correlations=False, correct_mag_bias=True):
    """Compute relative magnitudes between two events."""
    amplitudes, snrs_1, snrs_2 = relative_amplitude(
        st1, st2, event1, event2, noise_window, signal_window, min_snr, use_s_picks)

    compute_correlations = correlations is None
    if compute_correlations:
        correlations = {}

    relative_magnitudes = {}
    rmags_L2 = []
    rmags_dot = []

    for seed_id, amplitude_ratio in amplitudes.items():
        tr1 = st1.select(id=seed_id)[0]
        tr2 = st2.select(id=seed_id)[0]

        pick1 = _get_pick_for_station(event1, tr1.stats.station, use_s_picks)
        pick2 = _get_pick_for_station(event2, tr2.stats.station, use_s_picks)

        if pick1 is None or pick2 is None:
            continue

        if compute_correlations:
            cc = correlate(
                tr1.slice(starttime=pick1.time + signal_window[0],
                          endtime=pick1.time + signal_window[1]),
                tr2.slice(starttime=pick2.time + signal_window[0],
                          endtime=pick2.time + signal_window[1]),
                shift=int(shift * tr1.stats.sampling_rate)
            ).max()
            correlations[seed_id] = cc
        else:
            cc = correlations.get(seed_id, 0.0)

        if cc < min_cc:
            continue

        snr_x = snrs_1[seed_id]
        snr_y = snrs_2[seed_id]

        if not correct_mag_bias:
            cc = snr_x = snr_y = 1.0

        norm_ratio = np.linalg.norm(tr2.data) / np.linalg.norm(tr1.data)
        rel_mag = math.log10(norm_ratio) + math.log10(
            math.sqrt((1 + 1 / snr_y**2) / (1 + 1 / snr_x**2)) * cc)
        rmag_dot = math.log10(amplitude_ratio)

        relative_magnitudes[seed_id] = rel_mag
        rmags_L2.append(rel_mag)
        rmags_dot.append(rmag_dot)

    if return_correlations:
        return relative_magnitudes, correlations, rmags_L2, rmags_dot
    return relative_magnitudes, rmags_L2, rmags_dot

# -------------------------
# Main Execution
# -------------------------

# Load catalogs
rcat = pd.read_csv('reloc1D_ecat_coh_se2.5_clustered_rmag_test.csv')
icat = pd.read_csv('/Users/bnjo/home/bnjo/hoodF/util/hoodF_icat.csv')
icat['evids'] = [str(e) for e in icat['evids']]

# Filter
excluded_evids = ['10468778','10292693','10293323','10426898','10431753','10081343',
                  '10100138','10121223','10142768','10225238','10248448','10262938',
                  '10265193','10269303','10377068','10432263','10759013']

test = rcat.loc[~rcat['evids'].isin(excluded_evids)]
testT = test.loc[test['etype'] == 'icat']
testD = testT.copy()

waveforms_path = '/Users/bnjo/home/bnjo/hoodF/mags/waveforms/'
info = []

for _, row1 in tqdm(testT.iterrows(), total=testT.shape[0]):
    evid1 = row1['evids']
    mag1 = row1['mag']
    i = row1['eq_num']

    event1 = obspy.read_events(os.path.join(waveforms_path, f"{evid1}/*.xml"))[0]
    st1 = obspy.read(os.path.join(waveforms_path, f"{evid1}/*.mseed"))
    for ch in ["LH*", "VH*", "UH*"]:
        for tr in st1.select(channel=ch):
            st1.remove(tr)
    st1 = preprocess_stream(st1)

    for _, row2 in testD.iterrows():
        evid2 = row2['evids']
        if int(evid2) != int(evid1) and int(evid2) > int(evid1):
            mag2 = row2['mag']
            j = row2['eq_num']

            event2 = obspy.read_events(os.path.join(waveforms_path, f"{evid2}/*.xml"))[0]
            st2 = obspy.read(os.path.join(waveforms_path, f"{evid2}/*.mseed"))
            for ch in ["LH*", "VH*", "UH*"]:
                for tr in st2.select(channel=ch):
                    st2.remove(tr)
            st2 = preprocess_stream(st2)

            try:
                rmag, corr, rmags_L2, rmags_dot = relative_magnitude(
                    st1, st2, event1, event2,
                    noise_window=(-20, -1),
                    signal_window=(-0.5, 20),
                    min_snr=1.5, min_cc=0.5,
                    use_s_picks=False, correlations=None, shift=2.0,
                    return_correlations=True, correct_mag_bias=True
                )
                info.append([i, j, evid1, evid2, rmags_L2, rmags_dot,
                             np.mean(rmags_L2), np.mean(rmags_dot),
                             mag1, mag2, len(rmags_L2), len(st1), len(st2)])
            except Exception as e:
                print(f"Failed for {evid1}-{evid2}: {e}")

# Assemble results
idfF = pd.DataFrame(info, columns=[
    'i', 'j', 'evid1', 'evid2', 'dmag_L2', 'dmag_dot',
    'm_dmagL2', 'm_dmagDot', 'cat_mag1', 'cat_mag2', 'n_obs', 'n_trs1', 'n_trs2'
])
idfF.to_pickle('hoodF_1D_coh_rmag.pkl')
