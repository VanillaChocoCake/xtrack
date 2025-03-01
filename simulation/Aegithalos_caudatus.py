"""
Toolkit for Spectral Analysis, Filtering and Algorithm Implementation
Author: Sun Peihan
Date: 2025-02-28
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy import signal
from scipy.signal import savgol_filter, find_peaks
from scipy.signal.windows import hamming
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import bisect
import pickle
from scipy.interpolate import interp1d, CubicSpline, lagrange, UnivariateSpline
from collections import deque
import h5py

def read_pkl(filename: str) -> dict:
    """Load serialized Python objects from pickle file.

    Args:
        filename: Path to .pkl file

    Returns:
        Deserialized dictionary object

    Raises:
        FileNotFoundError: If specified file doesn't exist
        pickle.UnpicklingError: For corrupted pickle files
    """
    with open(filename, "rb") as f:
        dic = pickle.load(f)
    return dic

def load_matlab_v73(filename: str,
                    variable_name: str) -> np.ndarray:
    """Load MATLAB v7.3+ data files using HDF5 interface.

    Args:
        filename: Path to .mat file
        variable_name: Target dataset name

    Returns:
        Numpy array containing requested data

    Raises:
        KeyError: If specified variable not found
        OSError: For invalid file paths
    """
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        print("Available datasets:", keys)

        if variable_name in f:
            return np.array(f[variable_name])
        raise KeyError(f"Variable {variable_name} not found")

def interpolation(data: np.ndarray,
                  interpolate_method: str = "cubic",
                  interpolate_coef: float = 1) -> np.ndarray:
    """Perform 1D interpolation with various methods.

    Args:
        data: Input signal for interpolation
        interpolate_method: One of ['linear', 'cubic', 'lagrange', 'univariate']
        interpolate_coef: Upsampling factor (>=1)

    Returns:
        Interpolated signal

    Raises:
        ValueError: For unsupported interpolation methods
    """
    batch_size = len(data)
    if interpolate_method:
        freq = np.linspace(0, batch_size - 1, batch_size)
        freq_new = np.linspace(0, batch_size - 1, int(interpolate_coef * batch_size))

        method_map = {
            "linear": interp1d(freq, data),
            "cubic": CubicSpline(freq, data),
            "lagrange": lagrange(freq, data),
            "univariate": UnivariateSpline(freq, data, s=np.mean(data))
        }

        if interpolate_method not in method_map:
            raise ValueError(f"Invalid method: {interpolate_method}")

        return method_map[interpolate_method](freq_new)
    return data

def exclude_coherent_spectrum(tune_unit: np.ndarray,
                              spectrum: np.ndarray,
                              side_point_num: int,
                              threshold: float=0.4) -> np.ndarray:
    """Remove coherent spectral components via Gaussian substitution.

    Args:
        tune_unit: Normalized frequency axis
        spectrum: Power spectral density
        side_point_num: Points to exclude around peak
        threshold: Detection threshold (normalized)

    Returns:
        Modified spectrum with coherent components replaced
    """
    peak_idx = np.argmax(spectrum)
    peak_tune = tune_unit[peak_idx]
    left, right = find_closest_values(peak_tune, tune_unit)
    coherent_peak = left if np.abs(peak_tune - left) < np.abs(peak_tune - right) else right
    coherent_peak_index = np.where(tune_unit == coherent_peak)[0][0]

    # Normalize spectrum
    coef = 1/np.max(spectrum)
    spectrum *= coef

    # Define exclusion zone
    start_left = int(max(0, coherent_peak_index - side_point_num))
    stop_right = int(min(len(spectrum) - 1, coherent_peak_index + side_point_num))
    fit_mask = (spectrum <= threshold) & ((tune_unit >= tune_unit[start_left]) & (tune_unit <= tune_unit[stop_right]))
    fit_tune_unit = tune_unit[fit_mask]
    fit_spectrum = spectrum[fit_mask]

    # Gaussian substitution
    a, x0, phi = gaussian_peak_fit(fit_tune_unit, fit_spectrum)
    gaussian_fit = gaussian_function(tune_unit, a, x0, phi)
    new_mask = (spectrum > threshold) & ((tune_unit >= tune_unit[start_left]) & (tune_unit <= tune_unit[stop_right]))
    spectrum *= (spectrum <= threshold).astype(np.float64)
    spectrum += gaussian_fit*new_mask.astype(np.float64)
    # plt.figure()
    # plt.plot(tune_unit, scaled_spectrum, label="original")
    # plt.plot(tune_unit, spectrum, label="replaced")
    # plt.plot(tune_unit, gaussian_fit, label="fitted")
    # plt.legend()
    # plt.show()
    spectrum /= coef
    return spectrum

def exclude_frequency_component(t: np.ndarray,
                                y: np.ndarray,
                                frequency: float) -> np.ndarray:
    """
    Remove a specific frequency component from the input signal using harmonic fitting.

    Args:
        t: Time vector (1D array) representing the sampling points.
        y: Input signal values corresponding to the time vector (1D array).
        frequency: Target frequency to be removed from the signal (in Hz).

    Returns:
        Residual signal after removing the specified frequency component (1D array).
    """
    def harmonic_model(t, A, f, phi):
        """
        Harmonic signal model for fitting.

        Args:
            t: Time vector (1D array).
            A: Amplitude of the harmonic signal.
            f: Frequency of the harmonic signal (in Hz).
            phi: Phase offset of the harmonic signal (in radians).

        Returns:
            Harmonic signal values (1D array).
        """
        return A * np.sin(2 * np.pi * f * t + phi)

    # Initial parameter guesses for curve fitting
    p0 = [max(y), frequency, 0]

    # Perform curve fitting to estimate harmonic parameters
    params, _ = curve_fit(harmonic_model, xdata=t, ydata=y, p0=p0)

    # Extract fitted parameters
    A_fit, f_fit, phi_fit = params

    # Generate the fitted harmonic signal using estimated parameters
    fitted_harmonic = harmonic_model(t, A_fit, f_fit, phi_fit)

    # Subtract the fitted harmonic from the original signal
    residual_signal = y - fitted_harmonic
    return residual_signal

def exclude_coherent_signal(t: np.ndarray,
                            y: np.ndarray,
                            f_sampling: float,
                            freqs_range: np.ndarray[float, float]) -> np.ndarray:
    """
        Remove coherent frequency components from a time-domain signal using spectral analysis.

    Args:
        t: Time vector (1D array)
        y: Signal values corresponding to time vector (1D array)
        f_sampling: Sampling frequency in Hz
        freqs_range: Frequency range to analyze as (f_min, f_max). Use None for full spectrum.

    Returns:
        Filtered signal with coherent components removed (1D array)

    Notes:
        - Coherent components are identified as spectral peaks above 0.4 (normalized PSD)
        - Each detected coherent frequency is removed using harmonic subtraction
    """
    # Compute power spectral density
    freqs, psd = cal_psd(y, f_sampling)

    # Apply frequency range mask if specified
    if freqs_range is not [None, None]:
        mask = (freqs >= freqs_range[0]) & (freqs <= freqs_range[1])
    else:
        mask = freqs > 0

    # Filter frequency and PSD arrays
    freqs = freqs[mask]
    psd = psd[mask]

    # Normalize PSD to [0, 1] range
    psd = normalize_to_0_1(psd)

    # Detect spectral peaks and coherent components
    index_bool, index_value = find_local_maxima(psd)
    coherent_frequencies = psd >= 0.4 # Threshold for coherence
    frequencies_to_exclude_bool = index_bool & coherent_frequencies
    frequencies_to_exclude = freqs[frequencies_to_exclude_bool]

    # Remove each coherent frequency component
    for coherent_frequency in frequencies_to_exclude:
        y = exclude_frequency_component(t, y, coherent_frequency)
    return y

def covered_by_detector(central_frequency: float,
                        bandwidth: float,
                        sideband_width: float,
                        tune: float,
                        current_frequency: float) -> bool:
    """
    Determine if a given frequency range is covered by a detector's bandwidth and sidebands.

    Args:
        central_frequency: The center frequency of the detector's operating range (in Hz).
        bandwidth: The total bandwidth of the detector (in Hz).
        sideband_width: The width of the sidebands around the central frequency (in Hz).
        tune: The tuning parameter representing the offset from the harmonic (unitless).
        current_frequency: The frequency being evaluated for coverage (in Hz).

    Returns:
        bool: True if the frequency is covered by the detector's range, False otherwise.
    """
    # Calculate the lower and upper bounds of the detector's frequency range
    fl = central_frequency - bandwidth/2
    fh = central_frequency + bandwidth/2

    # Determine the nearest harmonic of the current frequency
    harmonic = np.round(central_frequency/current_frequency)

    # Check if the lower sideband is within the detector's range
    tune_covered_lower = (((harmonic - tune)*current_frequency - sideband_width/2 > fl) &
                          ((harmonic - tune)*current_frequency + sideband_width/2 < fh))

    # Check if the upper sideband is within the detector's range
    tune_covered_upper = (((harmonic + tune)*current_frequency - sideband_width/2 > fl) &
                          ((harmonic + tune)*current_frequency + sideband_width/2 < fh))

    # Return True if either the lower or upper sideband is covered
    return tune_covered_lower | tune_covered_upper

def covered_frequency_bands_minmax(central_frequency: float,
                                   bandwidth: float,
                                   sideband_width: float,
                                   tune_min: float, tune_max: float,
                                   start_frequency: float,
                                   end_frequency: float,
                                   step: float=0.01e6) -> (np.ndarray, np.ndarray):
    """
    Calculate the frequency bands and determine which bands are covered within the specified tuning range.

    Parameters:
    -----------
    central_frequency : float
        The central frequency of the detector.
    bandwidth : float
        The bandwidth of the detector.
    sideband_width : float
        The width of the sidebands to be considered.
    tune_min : float
        The minimum tuning offset from the central frequency.
    tune_max : float
        The maximum tuning offset from the central frequency.
    start_frequency : float
        The starting frequency of the range to analyze.
    end_frequency : float
        The ending frequency of the range to analyze.
    step : float, optional
        The step size for generating frequency bands (default is 0.01e6).

    Returns:
    --------
    frequency_bands : np.ndarray
        An array of frequency bands within the specified range.
    covered_bands : np.ndarray
        A boolean array indicating which frequency bands are covered within the tuning range.
    """
    # Define the lower and upper bounds of the frequency range
    fl = central_frequency - bandwidth/2
    fh = central_frequency + bandwidth/2

    # Generate an array of frequency bands within the specified range
    frequency_bands = np.linspace(start_frequency, end_frequency, num=int((end_frequency - start_frequency)/step + 1))

    # Calculate the nearest harmonic relationship between the central frequency and each frequency band
    harmonic = np.round(central_frequency/frequency_bands)

    # Check if the lower sideband (tune_min) is within the frequency range
    tune_min_covered_lower = (((harmonic - tune_min)*frequency_bands - sideband_width/2 > fl) &
                              ((harmonic - tune_min)*frequency_bands + sideband_width/2 < fh))

    # Check if the upper sideband (tune_min) is within the frequency range
    tune_min_covered_upper = (((harmonic + tune_min)*frequency_bands - sideband_width/2 > fl) &
                              ((harmonic + tune_min)*frequency_bands + sideband_width/2 < fh))

    # Check if the lower sideband (tune_max) is within the frequency range
    tune_max_covered_lower = (((harmonic - tune_max)*frequency_bands - sideband_width/2 > fl) &
                              ((harmonic - tune_max)*frequency_bands + sideband_width/2 < fh))

    # Check if the upper sideband (tune_max) is within the frequency range
    tune_max_covered_upper = (((harmonic + tune_max)*frequency_bands - sideband_width/2 > fl) &
                              ((harmonic + tune_max)*frequency_bands + sideband_width/2 < fh))

    # Combine the results for the lower and upper sidebands for tune_min and tune_max
    tune_min_covered = tune_min_covered_lower | tune_min_covered_upper
    tune_max_covered = tune_max_covered_lower | tune_max_covered_upper

    # Return the frequency bands and the boolean array indicating covered bands
    return frequency_bands, tune_min_covered & tune_max_covered

def plot_measured_results(dic: dict=None,
                          filename: str=None,
                          title: str="Comparison") -> None:
    """
    Plot the measured results for comparison, including nominal values, reference, predicted, and measured data.

    Parameters:
    -----------
    dic : dict, optional
        A dictionary containing the results to be plotted. If not provided, the function will attempt to load
        the data from a file.
    filename : str, optional
        The filename of a pickle file containing the results dictionary. Required if `dic` is not provided.
    title : str, optional
        The title of the plot, defaults to "Comparison" if not provided.
    Raises:
    -------
    ValueError
        If neither `dic` nor `filename` is provided.

    Returns:
    --------
    None
        Displays a plot of the results.
    """
    # Validate input: either a dictionary or a filename should be provided
    if dic is None and filename is None:
        raise ValueError("Either a dictionary or a filename should be provided")

    # Load the dictionary from the file if not provided
    if dic is None:
        with open(filename, "rb") as f:
            dic = pickle.load(f)
    qx = np.array(dic['qx'])
    failed_to_detect = np.array(dic['failed_to_detect'])
    q_ref_filtered = np.array(dic['q_ref_filtered'])
    q_predicted = np.array(dic['q_predicted'])
    q_measured_filtered = np.array(dic['q_measured_filtered'])
    # peak_detection = np.array(dic['peak_detection'])
    # cf = np.array(dic['curve_fitting'])
    # weight_ref = np.array(dic['weight_ref'])
    # weight_measured = np.array(dic['weight_measured'])
    plt.figure()
    plt.plot(qx, label='nominal')
    plt.plot(qx + 0.01, label='nominal + 0.01')
    plt.plot(qx - 0.01, label='nominal - 0.01')
    plt.plot(failed_to_detect, label='not covered area')
    plt.plot(q_ref_filtered, 'o', label='reference (filtered)', markersize=1)
    plt.plot(q_predicted, 'v', label='predicted', markersize=1)
    plt.plot(q_measured_filtered, 's', label='measured (filtered)', markersize=1)
    # plt.plot(peak_detection, '*', label='peak detection', markersize=1)
    # plt.plot(cf, 'p', label='curve fitting', markersize=1)
    # plt.plot(weight_ref, label='weight_ref')
    # plt.plot(weight_measured, label='weight_measured')
    plt.title(title)
    plt.legend()
    plt.show()

def find_closest_values(a: float,
                        b: np.ndarray) -> tuple:
    """
    Find the closest values in a sorted array `b` that are smaller and larger than the target value `a`.

    Parameters:
    -----------
    a : float
        The target value for which the closest values are to be found.
    b : np.ndarray
        A sorted array in ascending order.

    Returns:
    --------
    tuple : (lower_value, upper_value)
            - lower_value: The largest value in `b` that is smaller than `a`. Returns `None` if no such value exists.
            - upper_value: The smallest value in `b` that is larger than `a`. Returns `None` if no such value exists.
    """
    # Convert the NumPy array to a list for compatibility with the bisect module
    b = b.tolist()

    # Handle the case where the array is empty
    if len(b) == 0:
        return None, None

    # Find the insertion positions for the target value `a` in the sorted array `b`
    # Position to insert `a` to maintain sorted order (leftmost)
    left_pos = bisect.bisect_left(b, a)
    # Position to insert `a` to maintain sorted order (rightmost)
    right_pos = bisect.bisect_right(b, a)

    # Find the largest value smaller than `a`
    # Use the element before the insertion point
    lower = b[left_pos - 1] if left_pos > 0 else None

    # Find the smallest value larger than `a`
    # Use the element at the insertion point
    upper = b[right_pos] if right_pos < len(b) else None

    # Return the results as a tuple
    return lower, upper

def gaussian_function(x,
                      a,
                      x0,
                      sigma):
    """
    Compute the value of a Gaussian (normal) function at a given point `x`.

    The Gaussian function is defined as:
        f(x) = a * exp(-(x - x0)^2 / (2 * sigma^2))

    Parameters:
    -----------
    x : float or np.ndarray
        The point(s) at which to evaluate the Gaussian function.
    a : float
        The amplitude of the Gaussian function (maximum height of the peak).
    x0 : float
        The mean (center) of the Gaussian function.
    sigma : float
        The standard deviation (spread or width) of the Gaussian function.

    Returns:
    --------
    float or np.ndarray
        The value(s) of the Gaussian function evaluated at `x`.
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gaussian_peak_fit(x: np.ndarray,
                      y: np.ndarray) -> np.ndarray:
    """
    Fit a Gaussian function to the provided data points (x, y) and return the fitted parameters.

    The Gaussian function is defined as:
        f(x) = a * exp(-(x - x0)^2 / (2 * sigma^2))

    Parameters:
    -----------
    x : np.ndarray
        The x-coordinates of the data points.
    y : np.ndarray
        The y-coordinates of the data points.

    Returns:
    --------
    np.ndarray
        An array containing the fitted parameters [a, x0, sigma], where:
        - a: The amplitude of the Gaussian function.
        - x0: The mean (center) of the Gaussian function.
        - sigma: The standard deviation (spread) of the Gaussian function.
        If the fitting fails, the function returns the initial guess based on the maximum value in `y`.
    """
    # Initial parameter guesses
    max_idx = np.argmax(y)
    x0_guess = x[max_idx]
    a_guess = y[max_idx]
    # Guess for the standard deviation, assuming the data covers ~4σ range
    sigma_guess = (x[-1] - x[0]) / 2

    try:
        # Perform the curve fitting using the Gaussian function
        popt, _ = curve_fit(gaussian_function, x, y, p0=[a_guess, x0_guess, sigma_guess])
        # Return the fitted parameters [a, x0, sigma]
        return popt
    except:
        # Handle fitting failure by returning the initial guesses
        print("Fitting failed. Returning the initial guess based on the maximum value.")
        return np.array([a_guess, x0_guess, sigma_guess])

def generate_q_list(method: str,
                    n_points: int,
                    lower_limit: float,
                    upper_limit: float) -> np.ndarray:
    """
    Generate a list of values (`q_list`) based on the specified method, number of points, and limits.

    Parameters:
    -----------
    method : str
        The method used to generate the list. Supported methods are:
        - "sin": Generates a sinusoidal pattern.
        - "cos": Generates a cosine pattern.
        - "linear": Generates a linearly spaced list.
        - "random": Generates a randomized pattern based on the sinc function.
        - Any other value: Generates a constant list with the average of the limits.
    n_points : int
        The number of points to generate.
    lower_limit : float
        The lower bound of the generated values.
    upper_limit : float
        The upper bound of the generated values.

    Returns:
    --------
    np.ndarray
        An array of generated values (`q_list`) within the specified limits.
    """
    # Generate a linearly spaced array between 0 and 1 as the base for transformations
    x = np.linspace(0, 1, num=n_points)
    if method == "sin":
        # Generate a sinusoidal pattern
        q_list = (upper_limit - lower_limit) * 0.5 * np.sin(2 * np.pi * x)
        # Shift and scale the pattern to fit within the specified limits
        q_list = q_list - np.min(q_list) + lower_limit

    elif method == "cos":
        # Generate a cosine pattern
        q_list = (upper_limit - lower_limit) * 0.5 * np.cos(2 * np.pi * x)
        # Shift and scale the pattern to fit within the specified limits
        q_list = q_list - np.min(q_list) + lower_limit

    elif method == "linear":
        # Generate a linearly spaced list between the lower and upper limits
        q_list = np.linspace(lower_limit, upper_limit, num=n_points)

    elif method == "random":
        # Generate a randomized pattern using the sinc function
        y = np.sinc(6 * x)
        # Normalize and scale the pattern to fit within the specified limits
        y -= np.min(y)
        y *= (upper_limit - lower_limit) / np.max(y)
        y += lower_limit
        q_list = y

    else:
        # Default to a constant list with the average of the lower and upper limits
        q_list = (upper_limit + lower_limit) * 0.5 * np.ones_like(x)

    return q_list

def plot(x: np.ndarray,
         y: np.ndarray=None) -> None:
    """
    Generates a plot using Matplotlib. If only `x` is provided, it plots `x` against its index.
    If both `x` and `y` are provided, it plots `y` against `x`.

    Parameters:
    x (np.ndarray): The primary data array. If `y` is provided, `x` serves as the horizontal axis.
    y (np.ndarray, optional): The dependent variable array. Defaults to None.

    Returns:
    None
    """
    plt.figure()
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.show()

def cal_psd(data: np.ndarray,
            f_sampling: float = None) -> tuple:
    """
    Computes the Power Spectral Density (PSD) of the input signal using the Fast Fourier Transform (FFT).

    Parameters:
    data (np.ndarray): Input signal data, which can be one-dimensional or multi-dimensional.
    f_sampling (float, optional): Sampling frequency of the signal. If provided, the function returns frequency values.

    Returns:
    tuple: A tuple containing:
        - full_freqs (np.ndarray or None): Frequency axis values if `f_sampling` is provided; otherwise, None.
        - psd (np.ndarray): Computed Power Spectral Density (PSD) of the signal.
    """

    # Compute the FFT along the last axis
    psd = fft(data, axis=-1)

    # Shift zero-frequency component to the center of the spectrum
    psd = fftshift(psd, axes=-1)

    if f_sampling:
        # Compute frequency axis values when sampling frequency is provided
        full_freqs = fftshift(np.fft.fftfreq(data.shape[-1], 1 / f_sampling))

        # Compute the PSD using the squared magnitude of the FFT, normalized by the signal length and sampling frequency
        psd = np.abs(psd) ** 2 / (data.shape[-1] * f_sampling)
    else:
        # Compute the PSD without frequency scaling if sampling frequency is not provided
        psd = np.abs(psd) ** 2 / data.shape[-1]
        full_freqs = None

    # If the input data has multiple dimensions, sum the PSD across all dimensions except the last
    if data.ndim > 1:
        psd = np.sum(psd, axis=0)

    # Estimate the noise floor using the 5th percentile to enhance robustness
    noise_floor = np.percentile(psd, 5)

    # Subtract the estimated noise floor from the PSD
    psd -= noise_floor

    # Ensure the PSD does not contain negative values by clipping it to a minimum of zero
    psd = np.clip(psd, 0, None)

    return full_freqs, psd

def spectral_processing(x_data: np.ndarray,
                        batch_size: int,
                        f_sampling: float,
                        window_size: int,
                        f_rev: float,
                        tune_unit_lower_limit: float,
                        tune_unit_upper_limit: float,
                        smoothing_method: str = "gaussian",
                        interpolate_method: str = "cubic",
                        interpolate_coef: float = 1,
                        procedure: int = 1) -> tuple:
    """
    Processes spectral data using Fast Fourier Transform (FFT), folding, filtering,
    and interpolation techniques based on the selected procedure.

    Parameters:
    x_data (np.ndarray): Input time-domain signal, expected to be a 2D array of shape (num_batches, samples_per_batch).
    batch_size (int): Number of samples per batch.
    f_sampling (float): Sampling frequency of the input signal.
    window_size (int): Window size for smoothing operations.
    f_rev (float): Revolution frequency used for spectrum folding.
    tune_unit_lower_limit (float): Lower limit of the normalized tune unit.
    tune_unit_upper_limit (float): Upper limit of the normalized tune unit.
    smoothing_method (str, optional): Smoothing method to apply, either "gaussian" or "savgol". Defaults to "gaussian".
    interpolate_method (str, optional): Interpolation method, such as "cubic" or "univariate". Defaults to "cubic".
    interpolate_coef (float, optional): Scaling factor for interpolation. Defaults to 1.
    procedure (int, optional): Defines the processing order:
        - 1: Fold → Sum → Smoothing
        - 2: Fold → Smoothing → Sum
      Defaults to 1.

    Returns:
    tuple:
        - final_tune_unit (np.ndarray): Normalized frequency axis after processing.
        - final_psd (np.ndarray): Processed Power Spectral Density (PSD).
    """

    # ==================================================================
    # Input Validation and Preprocessing
    # ==================================================================
    if x_data.ndim != 2:
        raise ValueError("The input data must be a 2D array (num_batches, samples_per_batch).")

    num_batches, samples = x_data.shape
    if samples != batch_size:
        raise ValueError(f"Inconsistent batch size: {samples} vs {batch_size}")

    if interpolate_method is None:
        interpolate_coef = 1

    if procedure not in [1, 2]:
        raise ValueError('"procedure" should be either 1 (fold -> sum -> smoothing) or 2 (fold -> smoothing -> sum).')

    # ==================================================================
    # Precompute Global Parameters (Avoid Repetitive Computation in Loops)
    # ==================================================================
    # Generate full frequency axis (computed once)
    base_freqs, _ = fold_spectrum(np.empty(batch_size), f_rev, f_sampling)
    tune_unit = base_freqs / f_rev
    freq_mask = (tune_unit >= tune_unit_lower_limit) & (tune_unit <= tune_unit_upper_limit)
    final_tune_unit = tune_unit[freq_mask]

    # Preallocate memory for PSD storage
    psd_matrix = np.zeros((num_batches, len(tune_unit)), dtype=np.float64)

    # ==================================================================
    # Core Batch Processing (Vectorization + Memory Optimization)
    # ==================================================================
    # Compute FFT for each batch (vectorized for efficiency)
    spectra = fft(x_data, axis=1)
    spectra_shifted = fftshift(spectra, axes=1)

    # Compute Power Spectral Density (PSD) for each batch
    psd_all = np.abs(spectra_shifted) ** 2 / (batch_size * f_sampling)

    # Perform filtering and folding per batch (loop retained for memory efficiency)
    for i in range(num_batches):
        psd = psd_all[i]
        _, folded = fold_spectrum(psd, f_rev, f_sampling)

        if procedure == 1:
            # Fold → Sum → Smoothing
            psd_matrix[i] = folded
        elif procedure == 2:
            # Fold → Smoothing → Sum
            if smoothing_method == "gaussian":
                filtered = gaussian_filter(folded, window_size)
            else:
                filtered = savgol_filter(psd, window_size, 5)

            filtered = interpolation(filtered, "univariate", 1.0)
            psd_matrix[i] = filtered

    # ==================================================================
    # Post-Processing: Summation and Final Smoothing
    # ==================================================================
    if procedure == 1:
        # Fold → Sum → Smoothing
        final_psd = psd_matrix.sum(axis=0)
        if smoothing_method == "gaussian":
            final_psd = gaussian_filter(final_psd, window_size)
        else:
            final_psd = savgol_filter(final_psd, window_size, 5)

        final_psd = final_psd[freq_mask]
    elif procedure == 2:
        # Fold → Smoothing → Sum
        final_psd = psd_matrix[:, freq_mask].sum(axis=0)

    # Make PSD bigger than 0
    final_psd -= np.min(final_psd)
    final_psd *= 1e5

    # Apply interpolation to refine the final PSD
    final_psd = interpolation(final_psd, interpolate_method, interpolate_coef)

    # Generate final frequency axis after interpolation
    final_tune_unit = np.linspace(final_tune_unit[0], final_tune_unit[-1], num=len(final_psd))

    # ==================================================================
    # Baseline Correction (Using Percentile for Robust Estimation)
    # ==================================================================
    noise_floor = np.percentile(final_psd, 5)  # Use the 5th percentile for robust noise estimation
    final_psd -= noise_floor
    final_psd = np.clip(final_psd, 0, None)  # Ensure non-negative PSD values

    return final_tune_unit, final_psd

def windowed_reshape(arr: np.ndarray,
                     batch_size: int) -> np.ndarray:
    """
    Reshapes a one-dimensional array into batches and applies a Hamming window to each batch.

    Parameters:
    arr (np.ndarray): Input one-dimensional array.
    batch_size (int): Size of each batch.

    Returns:
    np.ndarray: A two-dimensional array (num_batches x batch_size) with the Hamming window applied to each batch.
    """

    # Compute the number of elements in the input array
    n = arr.size

    # Determine if zero-padding is needed to make the array length a multiple of batch_size
    remainder = n % batch_size
    if remainder != 0:
        pad_length = batch_size - remainder
        arr = np.concatenate([arr, np.zeros(pad_length, dtype=arr.dtype)])

    # Reshape the array into batches of size batch_size
    reshaped = arr.reshape(-1, batch_size)

    # Generate a Hamming window of the same size as each batch
    window = hamming(batch_size)

    # Apply the Hamming window to each batch and return the result
    return reshaped * window

def apply_bandpass_filter(x_data: np.ndarray,
                          fs: float,
                          lowcut: float,
                          highcut: float,
                          order: int = 5) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter to retain signals within a specified frequency range.

    Parameters:
    x_data (np.ndarray): Input signal data.
    fs (float): Sampling frequency in Hz.
    lowcut (float): Lower cutoff frequency of the bandpass filter in Hz.
    highcut (float): Upper cutoff frequency of the bandpass filter in Hz.
    order (int, optional): Order of the Butterworth filter. Defaults to 5.

    Returns:
    np.ndarray: Filtered signal data after bandpass filtering.
    """

    # Design a Butterworth bandpass filter with the specified cutoff frequencies and order
    b, a = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs)

    # Apply zero-phase filtering to avoid phase distortion
    filtered_data = signal.filtfilt(b, a, x_data)

    return filtered_data

def generate_noisy_signal(x_data: np.ndarray,
                          snr_db: float) -> tuple:
    """
    Generates Gaussian noise to achieve a specified signal-to-noise ratio (SNR)
    and applies it to the input signal.

    Parameters:
    x_data (np.ndarray): Original signal (one-dimensional array).
    snr_db (float): Target signal-to-noise ratio (SNR) in decibels (dB).

    Returns:
    tuple: (noisy_signal, noise)
        - noisy_signal (np.ndarray): Signal with added Gaussian noise.
        - noise (np.ndarray): Generated Gaussian noise.
    """

    # Generate Gaussian white noise with zero mean and standard deviation of 2.6
    noise = np.random.normal(loc=0, scale=2.6, size=len(x_data))

    # Compute the power of the original signal
    p_signal = np.mean(x_data ** 2)  # Power of the original signal

    # Compute the theoretical noise power based on the predefined standard deviation
    p_noise = 2.6 ** 2  # Noise theoretical power

    # Convert SNR from dB scale to a linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Compute the scaling factor to achieve the target SNR
    a = np.sqrt((snr_linear * p_noise) / p_signal)

    # Generate the noisy signal by scaling the original signal and adding noise
    noisy_signal = a * x_data + noise

    return noisy_signal, noise

def normalize_to_0_1(x_data: np.ndarray) -> np.ndarray:
    """
    Normalizes the input data to the range [0, 1].

    Parameters:
    x_data (list or np.ndarray): Input data, which can be a list or a NumPy array.

    Returns:
    np.ndarray: Normalized data with values in the range [0, 1].
    """

    # Convert input to a NumPy array to enable efficient mathematical operations
    x_data = np.array(x_data)

    # Compute the minimum and maximum values of the input data
    x_min = np.min(x_data)
    x_max = np.max(x_data)

    # Prevent division by zero in case all elements have the same value
    if x_max == x_min:
        return np.zeros_like(x_data)  # Return an array of zeros to indicate no variation in input data

    # Perform min-max normalization to scale data to the [0, 1] range
    normalized_data = (x_data - x_min) / (x_max - x_min)

    return normalized_data

def sgolay_filter(data: np.ndarray,
                  window_length: int,
                  polyorder: int = 4,
                  mode: str = 'mirror') -> np.ndarray:
    """
    Implements a Savitzky-Golay filter for smoothing a one-dimensional signal.

    Parameters:
    data (np.ndarray): Input one-dimensional array representing the signal to be filtered.
    window_length (int): Length of the filtering window (must be a positive odd integer).
    polyorder (int, optional): Order of the polynomial used for fitting (must be less than window_length). Default is 4.
    mode (str, optional): Boundary handling mode, which can be 'mirror', 'nearest', 'constant', or 'interp'. Default is 'mirror'.

    Returns:
    np.ndarray: Filtered data after applying the Savitzky-Golay filter.
    """

    # Validate input parameters
    if window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer.")

    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    # Apply the Savitzky-Golay filter to the input data
    return savgol_filter(data, window_length, polyorder, mode=mode)

def fold_spectrum(spectrum: np.ndarray,
                  f_rev: float,
                  f_sampling: float) -> tuple:
    """
    Fold a wideband PSD (Power Spectral Density) spectrum into the baseband range (0 ~ f_rev).

    Parameters:
    -----------
    spectrum : np.ndarray
        The power spectral density array after applying `fftshift`. The frequency range is (-f_sampling/2, f_sampling/2).
    f_rev : float
        The revolution frequency (baseband frequency).
    f_sampling : float
        The sampling rate, which must be an even multiple of `f_rev`.

    Returns:
    --------
    base_freqs : np.ndarray
        The frequency bins for the baseband (0 ~ f_rev).
    folded_psd : np.ndarray
        The folded power spectral density in the baseband range.
    """
    # Number of points in the spectrum
    n = len(spectrum)

    # Generate the frequency axis
    freqs = np.linspace(-f_sampling / 2, f_sampling / 2, n)

    # Calculate the number of frequency bands per side
    bands_per_side = int(2*f_sampling / (2 * f_rev))

    # Initialize the output array for the folded PSD
    folded_psd = []

    # Iterate over all frequency bands
    for band_idx in range(bands_per_side):
        # Skip all bands except the last one (for testing purposes)
        if band_idx != bands_per_side - 1:
            continue
        # Skip the baseband itself
        # if band_idx == 0:
        #     continue

        # Calculate the frequency range of the current band
        f_start = band_idx * f_rev/2
        f_end = (band_idx + 1) * f_rev/2

        # Get the bin indices for the current band
        band_mask = (freqs >= f_start) & (freqs < f_end)
        band_bins = np.where(band_mask)[0]

        # Skip empty bands (edge cases)
        if len(band_bins) == 0:
            continue

        # Extract the data for the current band
        band_data = spectrum[band_bins]

        # Handle odd-indexed bands by mirroring the data
        # Odd-indexed bands require mirroring
        if abs(band_idx) % 2 == 1:
            band_data = band_data[::-1]

        # Calculate the target index (corresponding position within the baseband)
        # target_slice = slice(0, len(band_data))

        # Add the data to the baseband
        # folded_psd[target_slice] += band_data
        folded_psd.append(band_data)

    # Pad the band data arrays to ensure equal length
    max_len = max(len(band_data) for band_data in folded_psd)
    padded_arrays = [np.pad(band_data, (0, max_len - len(band_data)), mode='constant') for band_data in folded_psd]

    # Sum the padded arrays to get the final folded PSD
    folded_psd = np.sum(padded_arrays, axis=0)

    # Generate the baseband frequency axis
    base_freqs = np.linspace(0, f_rev / 2, len(folded_psd), endpoint=False)
    return base_freqs, folded_psd

def gaussian_filter(x_data: np.ndarray,
                    window_size: int) -> np.ndarray:
    """
    Applies a Gaussian filter to a one-dimensional signal.
    :param x_data: The input signal (either a list or np.ndarray).
    :param window_size: The size of the filtering window (must be an odd integer).
    :return: The filtered signal (np.ndarray).
    """
    # Input validation
    if window_size < 3:
        raise ValueError("Window size must be ≥ 3")
    if window_size % 2 == 0:
        window_size += 1
        print(f"Warning: The window size has been automatically adjusted to an odd number: {window_size}")

    # Convert the input data to a numpy array
    x = np.asarray(x_data, dtype=np.float64)

    # Calculate Gaussian kernel parameters
    # Covers 95% of the energy
    truncate = 2.0
    # Standard deviation of the Gaussian kernel
    sigma = (window_size - 1) / 4

    # Apply the Gaussian filter (boundary handling mode can be adjusted)
    return gaussian_filter1d(x, sigma=sigma, truncate=truncate, mode='mirror')

def find_local_maxima(psd: np.ndarray) -> (list, list):
    """
    Identifies the local maxima in a power spectral density (PSD) signal.
    :param psd: The input power spectral density signal (np.ndarray).
    :return: A tuple containing:
        - A list of boolean values indicating the positions of the local maxima.
        - A list of indices corresponding to the positions of the local maxima.
    """
    # Convert the input PSD to a numpy array with float64 data type
    psd = np.asarray(psd, dtype=np.float64)

    # Initialize a boolean array to mark the positions of local maxima
    index_bool = np.zeros(len(psd), dtype=bool)

    # Identify the indices of the local maxima using the find_peaks function
    index_value, _ = find_peaks(psd)

    # Mark the positions of the local maxima in the boolean array
    index_bool[index_value] = True

    # Return the boolean array and the list of local maxima indices
    return index_bool.tolist(), index_value

def find_local_minima(psd: np.ndarray) -> (list, list):
    """
    Identifies the local minima in a power spectral density (PSD) signal.
    :param psd: The input power spectral density signal (np.ndarray).
    :return: A tuple containing:
        - A list of boolean values indicating the positions of the local minima.
        - A list of indices corresponding to the positions of the local minima.
    """
    # Convert the input PSD to a numpy array with float64 data type and negate the values
    psd = np.asarray(-psd, dtype=np.float64)

    # Initialize a boolean array to mark the positions of local minima
    index_bool = np.zeros(len(psd), dtype=bool)

    # Identify the indices of the local maxima in the negated signal (local minima in the original)
    index_value, _ = find_peaks(psd)

    # Mark the positions of the local minima in the boolean array
    index_bool[index_value] = True

    # Return the boolean array and the list of local minima indices
    return index_bool.tolist(), index_value

class EMA_PSD:
    def __init__(self, max_len, decay_factor=0.45):
        """
        Initializes the Exponential Moving Average (EMA) object with the given parameters.
        :param max_len: The maximum length of the signal data.
        :param decay_factor: The decay factor for the exponential moving average (default is 0.45).
        """
        self.max_len = max_len
        self.decay_factor = decay_factor
        self.first_append = True  # Flag to track the first append operation
        self.psd = 0  # Initialize the power spectral density (PSD) as 0
        self.tune_unit = []  # Initialize the tune unit list

    def append(self, tune_unit, psd):
        """
        Appends new power spectral density data and updates the EMA.
        :param tune_unit: The tuning unit associated with the PSD data.
        :param psd: The new power spectral density data to be included in the moving average.
        """
        # Initialize the PSD array during the first append
        if self.first_append:
            self.psd = np.zeros_like(psd)  # Initialize the PSD to a zero array of the same shape as psd
            self.first_append = False  # Set the flag to False after the first append

        # Apply the Exponential Moving Average (EMA) formula
        self.psd = self.decay_factor * self.psd + (1 - self.decay_factor) * psd

        # Update the tuning unit with the new value
        self.tune_unit = tune_unit

    def q_ref(self):
        """
        Returns the tuning unit corresponding to the maximum PSD value.
        :return: The tuning unit associated with the highest PSD value.
        """
        return self.tune_unit[np.argmax(self.psd)]  # Return the tune unit with the maximum PSD

class DualDetectorAdaptiveKalmanFilter:
    def __init__(
            self,
            initial_state=0.5,
            initial_estimate_error=1,
            process_noise=0.06,
            measurement_noise=2.6 ** 2,  # Initial measurement noise for both detectors
            max_len=8,
            alpha=0.4,
            min_weight=0
    ):
        """
        Initializes the Dual Detector Adaptive Kalman Filter with the given parameters.
        :param initial_state: The initial state estimate of the system.
        :param initial_estimate_error: The initial estimate of the error in the state estimate.
        :param process_noise: The process noise covariance, representing uncertainty in the system's dynamics.
        :param measurement_noise: The measurement noise covariance, assigned to both detectors.
        :param max_len: The maximum length of the residual history window for each detector.
        :param alpha: The smoothing factor for adjusting noise estimates based on the residual variance.
        :param min_weight: The minimum weight for the detector to prevent it from becoming too dominant.
        """
        # Initial state estimate and error covariance
        self.x = initial_state
        self.P = initial_estimate_error

        # Process noise covariance
        self.Q = process_noise

        # Measurement noise covariance for both detectors
        self.R1 = measurement_noise  # Initial measurement noise for detector 1
        self.R2 = measurement_noise  # Initial measurement noise for detector 2

        # Initial weights for the detectors (arbitrary large values)
        self.w1 = 114514
        self.w2 = 1919810

        # Residual history windows for each detector
        self.window1 = deque(maxlen=max_len)  # Residual history for detector 1
        self.window2 = deque(maxlen=max_len)  # Residual history for detector 2

        self.alpha = alpha  # Smoothing factor for noise adjustment
        self.min_weight = min_weight  # Minimum weight to prevent instability

    def _update_detector_noise(self, window, residual, current_noise):
        """
        Updates the measurement noise estimate for a single detector.
        When the number of residuals in the window is greater than or equal to 2,
        the variance of the residuals is used to update the noise estimate.
        :param window: The residual history window for the detector.
        :param residual: The current residual (difference between measurement and predicted state).
        :param current_noise: The current measurement noise estimate for the detector.
        :return: The updated measurement noise estimate.
        """
        window.append(residual)

        # If the window contains enough residuals, update the measurement noise estimate
        if len(window) >= 2:
            return self.alpha * np.var(window) + (1 - self.alpha) * current_noise

        # Otherwise, retain the current noise estimate
        return current_noise

    def _adjust_detector_weights(self):
        """
        Adjusts the weights of the detectors based on the current measurement noise estimates.
        Ensures that the weights remain within the minimum threshold to prevent instability.
        """
        total_precision = 1 / self.R1 + 1 / self.R2  # Total precision is the sum of the individual precisions

        # Calculate the normalized weight for each detector
        w1 = (1 / self.R1) / total_precision
        if w1 < self.min_weight:
            # If detector 1 weight is too small, adjust the weights accordingly
            w1 = self.min_weight
            w2 = 1 - w1
            self.R1 = 1 / (w1 * total_precision)
            self.R2 = 1 / (w2 * total_precision)
        elif w1 > 1 - self.min_weight:
            # If detector 1 weight is too large, adjust the weights accordingly
            w1 = 1 - self.min_weight
            w2 = 1 - w1
            self.R1 = 1 / (w1 * total_precision)
            self.R2 = 1 / (w2 * total_precision)

    def predict_update(self, z1, z2):
        """
        Performs the prediction and update steps of the Kalman filter using measurements from two detectors.
        :param z1: The measurement from detector 1.
        :param z2: The measurement from detector 2.
        :return: The updated state estimate.
        """
        # ----------- Prediction Step -----------
        # Predict the next state based on the current state estimate
        x_pred = self.x

        # Predict the error covariance, considering process noise
        P_pred = self.P + self.Q

        # ----------- Measurement Fusion Step -----------
        # Calculate the total precision (sum of individual detector precisions)
        total_precision = 1 / self.R1 + 1 / self.R2

        # Normalize the weights for each detector based on their precision
        self.w1 = (1 / self.R1) / total_precision
        self.w2 = (1 / self.R2) / total_precision

        # Fuse the measurements from both detectors using the calculated weights
        z_fused = self.w1 * z1 + self.w2 * z2

        # Calculate the equivalent measurement noise after fusion
        R_fused = 1 / total_precision

        # ----------- Update Step -----------
        # Compute the Kalman gain based on predicted error covariance and measurement noise
        K = P_pred / (P_pred + R_fused)

        # Compute the fused residual (innovation)
        residual_fused = z_fused - x_pred

        # Update the state estimate using the Kalman gain and residual
        self.x = x_pred + K * residual_fused

        # Update the error covariance
        self.P = (1 - K) * P_pred

        # ----------- Noise Update -----------
        # Calculate the residuals for both detectors and update the noise estimates
        residual1 = z1 - x_pred
        residual2 = z2 - x_pred
        self.R1 = self._update_detector_noise(self.window1, residual1, self.R1)
        self.R2 = self._update_detector_noise(self.window2, residual2, self.R2)

        # Adjust the detector weights and balance the measurement noise
        self._adjust_detector_weights()

        # Update the process noise based on the residual of the fused measurement
        self.Q = self.alpha * abs(residual_fused) + (1 - self.alpha) * self.Q

        return self.x

    def detector_weights(self):
        """
        Returns the normalized weights of the two detectors.
        The weights are guaranteed to sum to 1.
        :return: The normalized weights of detector 1 and detector 2.
        """
        total = self.w1 + self.w2
        return self.w1 / total, self.w2 / total  # Normalize weights to ensure they sum to 1

class MADFilter:
    def __init__(self, window_size=10, threshold=2):
        """
        Traditional MAD-based outlier detector with sliding window
        :param window_size: Size of the data window for MAD calculation
        :param threshold: Threshold multiplier for MAD-based detection
        """
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = []
        self.filtered_values = []

    def _mad(self, data):
        """Helper function to compute MAD"""
        median = np.median(data)
        abs_dev = np.abs(data - median)
        return np.median(abs_dev) * 1.4826  # Scaled MAD

    def process(self, new_value):
        """
        Process new data point with traditional MAD method
        :param new_value: New measurement value
        :return: Filtered value (outliers replaced with window median)
        """
        # Maintain sliding window
        self.data_window.append(new_value)
        if len(self.data_window) > self.window_size:
            self.data_window.pop(0)

        # Calculate statistics
        if len(self.data_window) >= self.window_size//2:  # Minimum samples for meaningful MAD
            current_median = np.median(self.data_window)
            mad = self._mad(np.array(self.data_window))
            upper_bound = current_median + self.threshold * mad
            lower_bound = current_median - self.threshold * mad

            # Replace outliers with median
            filtered = new_value if (lower_bound <= new_value <= upper_bound) else current_median
        else:
            filtered = new_value  # Pass through initially

        self.filtered_values.append(filtered)
        return filtered
