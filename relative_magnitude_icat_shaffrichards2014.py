# Standard library imports
import os
import math
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ObsPy imports
import obspy
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.event import Catalog
from obspy.signal.cross_correlation import xcorr_pick_correction, correlate

# Set up ObsPy FDSN client
client = Client("IRIS")

# -------------------
# Helper Functions
# -------------------

def preprocess_stream(st, freqmin=2.0, freqmax=10.0):
    """
    Preprocess a seismic Stream: detrend, demean, and bandpass filter.
    """
    st.detrend("demean")
    st.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
    return st

def filter_picks(
    catalog,
    stations=None,
    channels=None,
    networks=None,
    locations=None,
    top_n_picks=None,
    evaluation_mode="all"
):
    """
    Filter picks in a Catalog based on station, channel, network, location, or evaluation mode.
    """
    filtered_catalog = catalog.copy()

    if stations:
        for event in filtered_catalog:
            event.picks = [pick for pick in event.picks if pick.waveform_id.station_code in stations]
    if channels:
        for event in filtered_catalog:
            event.picks = [pick for pick in event.picks if pick.waveform_id.channel_code in channels]
    if networks:
        for event in filtered_catalog:
            event.picks = [pick for pick in event.picks if pick.waveform_id.network_code in networks]
    if locations:
        for event in filtered_catalog:
            event.picks = [pick for pick in event.picks if pick.waveform_id.location_code in locations]
    if evaluation_mode == 'manual':
        for event in filtered_catalog:
            event.picks = [pick for pick in event.picks if pick.evaluation_mode == 'manual']
    elif evaluation_mode == 'automatic':
        for event in filtered_catalog:
            event.picks = [pick for pick in event.picks if pick.evaluation_mode == 'automatic']

    if top_n_picks:
        all_picks = []
        for event in filtered_catalog:
            all_picks += [(pick.waveform_id.station_code, pick.waveform_id.channel_code) for pick in event.picks]
        counted = Counter(all_picks).most_common()
        all_picks = [item[0] for item in counted[:top_n_picks]]
        for event in filtered_catalog:
            event.picks = [pick for pick in event.picks if (pick.waveform_id.station_code, pick.waveform_id.channel_code) in all_picks]

    tmp_catalog = Catalog()
    for event in filtered_catalog:
        if event.picks:
            tmp_catalog.append(event)

    return tmp_catalog

# Other helper functions (_get_pick_for_station, _rms, _snr, _get_signal_and_noise, etc.) would follow similarly, cleaned up.

# -------------------
# Load Catalogs
# -------------------

rcat = pd.read_csv('reloc1D_ecat_coh_se2.5_clustered_rmag_test.csv')
icat = pd.read_csv('/Users/bnjo/home/bnjo/hoodF/util/hoodF_icat.csv')

etype = ['icat' if len(e) == 8 else 'ecat' for e in rcat['evids'].unique()]
rcat['etype'] = etype

icat['evid'] = icat['evids'].values
icat['evids'] = [str(e) for e in icat['evid'].values]

# -------------------
# Analysis Block
# -------------------

test = rcat.loc[~rcat['evids'].isin(['10468778', '10292693', '10293323', '10426898',
                                    '10431753', '10081343', '10100138', '10121223',
                                    '10142768', '10225238', '10248448', '10262938',
                                    '10265193', '10269303', '10377068', '10432263',
                                    '10759013'])]

testT = test.loc[test['etype'] == 'icat']
testD = testT

waveforms_path = '/Users/bnjo/home/bnjo/hoodF/mags/waveforms/'
info = []

for index, row in tqdm(testT.iterrows(), total=testT.shape[0]):
    evid1 = row.evids
    mag1 = row.mag
    i = row.eq_num
    event1 = obspy.read_events(os.path.join(waveforms_path, f"{str(evid1)}/*.xml"))[0]
    st1 = obspy.read(os.path.join(waveforms_path, f"{str(evid1)}/*.mseed"))

    for ch in ["LH*", "VH*", "UH*"]:
        for tr in st1.select(channel=ch):
            st1.remove(tr)
    st1 = preprocess_stream(st1)

    for idx, row2 in testD.iterrows():
        evid2 = row2.evids
        if int(evid2) != int(evid1) and int(evid2) > int(evid1):
            mag2 = row2.mag
            j = row2.eq_num
            event2 = obspy.read_events(os.path.join(waveforms_path, f"{str(evid2)}/*xml"))[0]
            st2 = obspy.read(os.path.join(waveforms_path, f"{str(evid2)}/*mseed"))

            for ch in ["LH*", "VH*", "UH*"]:
                for tr in st2.select(channel=ch):
                    st2.remove(tr)
            st2 = preprocess_stream(st2)

            try:
                rmag, corr, rmags_L2, rmags_dot = relative_magnitude(
                    st1, st2, event1, event2,
                    noise_window=(-20, -1), signal_window=(-0.5, 20),
                    min_snr=1.50, min_cc=0.5,
                    use_s_picks=False, correlations=None, shift=2.00,
                    return_correlations=True, correct_mag_bias=True
                )
                info.append([i, j, evid1, evid2, rmags_L2, rmags_dot,
                             np.mean(rmags_L2), np.mean(rmags_dot),
                             mag1, mag2, len(rmags_L2), len(st1), len(st2)])
            except Exception as e:
                print(f'{evid1} and {evid2} failed: {e}')

idfF = pd.DataFrame(info, columns=['i', 'j', 'evid1', 'evid2',
                                   'dmag_L2', 'dmag_dot',
                                   'm_dmagL2', 'm_dmagDot',
                                   'cat_mag1', 'cat_mag2',
                                   'n_obs', 'n_trs1', 'n_trs2'])

idfF = idfF.dropna()
idfF.to_pickle('hoodF_1D_coh_rmag.pkl')

# -------------------
# Plotting Results
# -------------------

# Scatter plot: Relative Magnitudes
plt.figure(figsize=(8, 8))
plt.scatter(idfF['m_dmagL2'], idfF['m_dmagDot'], color='indigo', s=1, marker='o', label='Event Pairs')

min_val = min(idfF['m_dmagL2'].min(), idfF['m_dmagDot'].min())
max_val = max(idfF['m_dmagL2'].max(), idfF['m_dmagDot'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

plt.xlabel("Magnitude Diff: δmag1 = log(α)")
plt.ylabel("Magnitude Diff: δmag2 = log(∥ỹ∥/∥x̃∥) + bias")
plt.title("Comparison of Relative Magnitude Methods")
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot: Catalog vs New Magnitudes (this would need defining cmag columns)
# Future note: cmag1-dot, cmag1-L2 columns must be created first before plotting here!
