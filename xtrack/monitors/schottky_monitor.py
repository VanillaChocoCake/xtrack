import numpy as np
import scipy as sp
from scipy.constants import c


class SchottkyMonitor():
    def __init__(self, f_rev, schottky_harmonic, n_taylor):
        """
        Tracking element computing Schottky spectra 
        Equations based on JINST 19 P03017, C.lannoy  and al.

        Parameters
        ----------
        f_rev : float
                Revolution frequency.

        schottky_harmonic : int
                Harmonic of the Schottky monitor.

        n_taylor : int
                Number of term used for the Taylor expansion (4 is enough for LHC conditions).
        """

        self.f_rev = f_rev
        if n_taylor < 1:
            raise ValueError('At least one coefficient for the Taylor expansion is needed')
        self.n_taylor = n_taylor
        # Taylor expantion around central Schottky frequency omega_c
        self.omega_c = 2 * np.pi * f_rev * schottky_harmonic
        self.x_coeff, self.y_coeff, self.z_coeff = [], [], []
        self.initialised_with_first_tracking = False

    def track(self, particles):
        mask_alive = particles.state > 0
        tau = -particles.zeta[mask_alive] / (c * particles.beta0[mask_alive])

        # If first time calling the function, store bunch parameters for 
        # computing the upper bound of the Taylor approximation
        if not self.initialised_with_first_tracking:
            self.tau_max = 4 * np.std(tau)
            self.x_max = 4 * np.std(particles.x[mask_alive])
            self.y_max = 4 * np.std(particles.y[mask_alive])
            self.N_macropart_max = len(tau)
            self.initialised_with_first_tracking = True

        # Calculates the longitudinal and transverse coefficients (L and T) as defined in Eqs (2.2) and (2.4)
        z_terms = np.empty((self.n_taylor, len(tau)), dtype=np.csingle)
        z_terms[0, :] = np.exp(1j * self.omega_c * tau)
        for l in range(1, self.n_taylor):
            z_terms[l, :] = z_terms[l - 1, :] * tau

        self.x_coeff.append(np.sum(z_terms * particles.x[mask_alive], axis=1))
        self.y_coeff.append(np.sum(z_terms * particles.y[mask_alive], axis=1))
        self.z_coeff.append(np.sum(z_terms, axis=1))

    def process_spectrum(self, inst_spectrum_len, deltaQ, band_width, Qx, Qy,
                         x=True, y=True, z=True, flattop_window=True):
        """
        Compute Schottky spectra from the stored longitudinal and transverse coefficients

        Parameters
        ----------
        inst_spectrum_len : int
                Number of revolution used to compute a single instataneous spectra.
                
        deltaQ : float 
                Frequency resolution of the spectra (in tune unit).

        band_width : float
                Range of frequency (in tune unit) where the spectrum will be computed for each band, 
                should be between 0 and 1.

        Qx, Qy : float
                Transverse tunes used to set the central frequency around which the 
                transverse side-bands will be computed. 

        x, y, z: bool
                Which band are to be computed. x, y, z stand for, respectively, 
                transverse horizontal, transverse vertical and longitudinal bands.

        flattop_window: bool
                Multiply time signal by flattop window, else no windowing used (=rectangular window).
        """

        if inst_spectrum_len > len(self.x_coeff):
            raise ValueError(f'Not enough turns tracked to produce a single instataneous spectra \n \
                               Number of turns tracked: {len(self.x_coeff)} \n \
                               Length of instataneous spectra: {inst_spectrum_len}')
        if band_width < 0 or band_width > 1:
            raise ValueError('Band_width should be expressed in tune unit and between 0 and 1')

        # If it's the first time calling the method we need to initialise it.
        if not hasattr(self, 'processing_param'):
            self.processing_param = locals()
            self._init_processing(deltaQ, Qx, Qy, band_width)

        # Not the first time calling this method, we will append the new instantaneous Schottky PSDs to the 
        # existing ones. In this case we need to confirm that the processing parameters are identical.
        elif any(self.processing_param[key] != value
                 for key, value in locals().items() if key not in ['x', 'y', 'z']):
            raise ValueError('Different parameters for the processing, ' +
                             'keep the same parameters (exept x, y and z) or use "clear_spectrum()". \n' +
                             'Existing parameters:' + str(self.processing_param) + '\n New parameters:' + str(locals()))
        if flattop_window:
            window = sp.signal.windows.flattop(inst_spectrum_len)
        else:
            window = np.ones(inst_spectrum_len)
        window /= np.sum(window)  # Normalising window

        region_to_process = []
        if x:
            region_to_process.extend(['lowerH', 'upperH'])
        if y:
            region_to_process.extend(['lowerV', 'upperV'])
        if z:
            region_to_process.extend(['center'])

        for region in region_to_process:
            freq = self.frequencies[region]
            if region == 'center':
                coeff = self.z_coeff
            elif region == 'lowerH' or region == 'upperH':
                coeff = self.x_coeff
            elif region == 'lowerV' or region == 'upperV':
                coeff = self.y_coeff
            else:
                raise ValueError('Frequency region not defined:' + region)

            # Computing instataneous Schottky spectra as defined in Eqs. (2.2) and (2.4)
            n_freq = len(freq)
            delta_omega = freq * 2 * np.pi * self.f_rev
            alpha = np.empty((self.n_taylor, n_freq), dtype=np.csingle)
            alpha[0, :] = np.ones(n_freq)
            for l in range(1, self.n_taylor):
                alpha[l, :] = alpha[l - 1, :] * 1j * delta_omega / l
            first_exponential = (np.vander(np.exp(1j * delta_omega / self.f_rev),
                                           N=inst_spectrum_len, increasing=True) * window).T
            n_inst_spectra = len(coeff) // inst_spectrum_len
            for i in range(len(self.instantaneous_PSDs[region]), n_inst_spectra):
                print(f'Processing {region} Schottky spectrum {i + 1}/{n_inst_spectra}', end='\r')
                # Seclecting the coefficient (x, y, or z) needed to calculate the i^th instantaneous Schottky spectra
                inst_coeff = np.array(coeff[i * inst_spectrum_len:(i + 1) * inst_spectrum_len])
                spectrum = np.sum(np.dot(inst_coeff, alpha) * first_exponential, axis=0)
                self.instantaneous_PSDs[region].append(abs(spectrum) ** 2 / self.N_macropart_max)
            self.PSD_avg[region] = np.mean(self.instantaneous_PSDs[region], axis=0)
            print(f'{region} band of Schottky spectrum processed')
        self._check_Taylor_approx()

    def _init_processing(self, deltaQ, Qx, Qy, band_width):
        '''
        For each region of the Schottky spectrum, create an array of normalised frequencies 
        from -band_with/2 to +band_with/2 around the center of the Schottky band. 
        '''
        n_freq = band_width / deltaQ
        center_freq = np.arange(-(n_freq // 2), (n_freq) // 2) * band_width / n_freq
        self.frequencies = {
            'lowerH': center_freq - (Qx % 1),
            'upperH': center_freq + (Qx % 1),
            'lowerV': center_freq - (Qy % 1),
            'upperV': center_freq + (Qy % 1),
            'center': center_freq
        }
        # Create dic where the instataneous and averaged PSDs will be stored
        self.instantaneous_PSDs = {i: [] for i in self.frequencies.keys()}
        self.PSD_avg = {i: [] for i in self.frequencies.keys()}

    def _check_Taylor_approx(self):
        #Longitudinal band, Eq. (2.3)
        if self.processing_param['z']:
            delta_omega_max = max(self.frequencies['center']) * 2 * np.pi * self.f_rev
            max_error = self.N_macropart_max ** 0.5 * (delta_omega_max * self.tau_max) ** self.n_taylor * \
                        np.exp(delta_omega_max * self.tau_max) / sp.special.factorial(self.n_taylor, exact=True)
            if np.sqrt(self.PSD_avg['center'][0]) < 100 * max_error:
                print('Number of Taylor terms too low for the longitudinal band')
            print(f'Maximal Talor truncation error in z plane to be compared against sqrt(PSD): {max_error}')

        #transverse bands
        if self.processing_param['x']:
            delta_omega_max = max(self.frequencies['upperH']) * 2 * np.pi * self.f_rev
            max_error = self.N_macropart_max ** 0.5 * self.x_max * (delta_omega_max * self.tau_max) ** self.n_taylor * \
                        np.exp(delta_omega_max * self.tau_max) / sp.special.factorial(self.n_taylor, exact=True)
            if np.sqrt(self.PSD_avg['upperH'][0]) < 100 * max_error:
                print('Number of Taylor terms too low for the horizontal bands')
            print(f'Maximal Talor truncation error in x plane to be compared against sqrt(PSD): {max_error}')
        if self.processing_param['y']:
            delta_omega_max = max(self.frequencies['upperV']) * 2 * np.pi * self.f_rev
            max_error = self.N_macropart_max ** 0.5 * self.y_max * (delta_omega_max * self.tau_max) ** self.n_taylor * \
                        np.exp(delta_omega_max * self.tau_max) / sp.special.factorial(self.n_taylor, exact=True)
            if np.sqrt(self.PSD_avg['upperV'][0]) < 100 * max_error:
                print('Number of Taylor terms too low for the vertical bands')
            print(f'Maximal Talor truncation error in y plane to be compared against sqrt(PSD): {max_error}')

    def clear_spectrum(self):
        """
        Clear the instantaneous spectra but keep the coefficients L and T.
        Can be use to recompute Schottky spectra for different processing 
        parameters (window, frequency resolution, band widths) without 
        tracking the particles agan
        """
        if hasattr(self, 'processing_param'):
            delattr(self, 'processing_param')
            delattr(self, 'frequencies')
            delattr(self, 'instantaneous_PSDs')
            delattr(self, 'PSD_avg')

    def clear_all(self):
        """
        Reinitialise monitor
        """
        self.clear_spectrum()
        self.x_coeff = []
        self.y_coeff = []
        self.z_coeff = []
