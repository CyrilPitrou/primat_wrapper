# -*- coding: utf-8 -*-
"""
PyPR_nuclear_data.py
====================
Loads all nuclear rate tables from disk and exposes them as attributes of a
``NuclearData`` instance.  This is the *only* module that performs nuclear-rate
file I/O, replacing the file-loading code that was previously scattered across
``PyPR_init.py``.

Usage::

    from PyPR.PyPR_nuclear_data import NuclearData
    from PyPR.PyPR_config import PyPRConfig

    cfg  = PyPRConfig(params)
    data = NuclearData(cfg)          # loads files once
    # data.npdg_T9, data.npdg_median, data.npdg_expsigma, ...

All tables are stored as plain NumPy arrays.  No interpolation is performed
here; that is left to the nuclear network classes.
"""

import os
import numpy as np


class NuclearData:
    """
    Loads and stores all nuclear reaction rate tables.

    Parameters
    ----------
    cfg : PyPRConfig
        A fully constructed configuration object.  Used only for
        ``cfg.working_dir`` and ``cfg.rates_dir``.
    """

    def __init__(self, cfg):
        wr  = cfg.working_dir
        rd  = cfg.rates_dir                                   # e.g. "key_primat_rates/"
        key = os.path.join(wr, "Rates", "nuclear", rd)
        oth = os.path.join(wr, "Rates", "nuclear", "other_nucl_rates", "")

        if cfg.verbose_flag:
            print("[rates] Loading nuclear rate tables from", key)

        # ----------------------------------------------------------------
        # Detailed-balance coefficients for the 12 fundamental reactions
        # ----------------------------------------------------------------
        # np -> dg
        self.alpha_npdg, self.beta_npdg, self.gamma_npdg = 4.71614e+09, 1.5, -25.815
        self.npdg_T9, self.npdg_median, self.npdg_expsigma = np.loadtxt(key + "npdg.txt", unpack=True)
        # dp -> He3g
        self.alpha_dpHe3g, self.beta_dpHe3g, self.gamma_dpHe3g = 1.6335e+10, 1.5, -63.7491
        self.dpHe3g_T9, self.dpHe3g_median, self.dpHe3g_expsigma = np.loadtxt(key + "dpHe3g.txt", unpack=True)
        # dd -> He3n
        self.alpha_ddHe3n, self.beta_ddHe3n, self.gamma_ddHe3n = 1.73183e+00, 0., -37.9341
        self.ddHe3n_T9, self.ddHe3n_median, self.ddHe3n_expsigma = np.loadtxt(key + "ddHe3n.txt", unpack=True)
        # dd -> tp
        self.alpha_ddtp, self.beta_ddtp, self.gamma_ddtp = 1.73492e+00, 0., -46.7971
        self.ddtp_T9, self.ddtp_median, self.ddtp_expsigma = np.loadtxt(key + "ddtp.txt", unpack=True)
        # tp -> ag
        self.alpha_tpag, self.beta_tpag, self.gamma_tpag = 2.61058e+10, 1.5, -229.93
        self.tpag_T9, self.tpag_median, self.tpag_expsigma = np.loadtxt(key + "tpag.txt", unpack=True)
        # td -> an
        self.alpha_tdan, self.beta_tdan, self.gamma_tdan = 5.5369e+00, 0., -204.1236
        self.tdan_T9, self.tdan_median, self.tdan_expsigma = np.loadtxt(key + "tdan.txt", unpack=True)
        # ta -> Li7g
        self.alpha_taLi7g, self.beta_taLi7g, self.gamma_taLi7g = 1.1133e+10, 1.5, -28.6355
        self.taLi7g_T9, self.taLi7g_median, self.taLi7g_expsigma = np.loadtxt(key + "taLi7g.txt", unpack=True)
        # He3n -> tp
        self.alpha_He3ntp, self.beta_He3ntp, self.gamma_He3ntp = 1.00178e+00, 0.0, -8.8630
        self.He3ntp_T9, self.He3ntp_median, self.He3ntp_expsigma = np.loadtxt(key + "He3ntp.txt", unpack=True)
        # He3d -> ap
        self.alpha_He3dap, self.beta_He3dap, self.gamma_He3dap = 5.5438e+00, 0.0, -212.987
        self.He3dap_T9, self.He3dap_median, self.He3dap_expsigma = np.loadtxt(key + "He3dap.txt", unpack=True)
        # He3a -> Be7g
        self.alpha_He3aBe7g, self.beta_He3aBe7g, self.gamma_He3aBe7g = 1.11289e+10, 1.5, -18.4179
        self.He3aBe7g_T9, self.He3aBe7g_median, self.He3aBe7g_expsigma = np.loadtxt(key + "He3aBe7g.txt", unpack=True)
        # Be7n -> Li7p
        self.alpha_Be7nLi7p, self.beta_Be7nLi7p, self.gamma_Be7nLi7p = 1.00215, 0., -19.0806
        self.Be7nLi7p_T9, self.Be7nLi7p_median, self.Be7nLi7p_expsigma = np.loadtxt(key + "Be7nLi7p.txt", unpack=True)
        # Li7p -> aa
        self.alpha_Li7paa, self.beta_Li7paa, self.gamma_Li7paa = 4.6898, 0., -201.295
        self.Li7paa_T9, self.Li7paa_median, self.Li7paa_expsigma = np.loadtxt(key + "Li7paa.txt", unpack=True)

        if cfg.verbose_flag:
            print("[rates] All 12 key rate tables loaded.")

        if cfg.smallnet_flag:
            return

        # ----------------------------------------------------------------
        # Additional 51 reactions (full 63-reaction network)
        # ----------------------------------------------------------------
        # Li7p -> aag
        self.alpha_Li7paag, self.beta_Li7paag, self.gamma_Li7paag = 4.6898, 0., -201.295
        self.Li7paag_T9, self.Li7paag_median, self.Li7paag_expsigma = np.loadtxt(oth + "Li7paag.txt", unpack=True)
        # Be7n -> aa
        self.alpha_Be7naa, self.beta_Be7naa, self.gamma_Be7naa = 4.6982, 0., -220.3871
        self.Be7naa_T9, self.Be7naa_median, self.Be7naa_expsigma = np.loadtxt(oth + "Be7naa.txt", unpack=True)
        # Be7d -> aap
        self.alpha_Be7daap, self.beta_Be7daap, self.gamma_Be7daap = 9.9579e-10, -1.5, -194.5722
        self.Be7daap_T9, self.Be7daap_median, self.Be7daap_expsigma = np.loadtxt(oth + "Be7daap.txt", unpack=True)
        # da -> Li6g
        self.alpha_daLi6g, self.beta_daLi6g, self.gamma_daLi6g = 1.53053e+10, 1.5, -17.1023
        self.daLi6g_T9, self.daLi6g_median, self.daLi6g_expsigma = np.loadtxt(oth + "daLi6g.txt", unpack=True)
        # Li6p -> Be7g
        self.alpha_Li6pBe7g, self.beta_Li6pBe7g, self.gamma_Li6pBe7g = 1.18778e+10, 1.5, -65.0648
        self.Li6pBe7g_T9, self.Li6pBe7g_median, self.Li6pBe7g_expsigma = np.loadtxt(oth + "Li6pBe7g.txt", unpack=True)
        # Li6p -> He3a
        self.alpha_Li6pHe3a, self.beta_Li6pHe3a, self.gamma_Li6pHe3a = 1.06729, 0., -46.6469
        self.Li6pHe3a_T9, self.Li6pHe3a_median, self.Li6pHe3a_expsigma = np.loadtxt(oth + "Li6pHe3a.txt", unpack=True)
        # B8n -> aap
        self.alpha_B8naap, self.beta_B8naap, self.gamma_B8naap = 3.6007e-10, -1.5, -218.7915
        self.B8naap_T9, self.B8naap_median, self.B8naap_expsigma = np.loadtxt(oth + "B8naap.txt", unpack=True)
        # Li6He3 -> aap
        self.alpha_Li6He3aap, self.beta_Li6He3aap, self.gamma_Li6He3aap = 7.2413e-10, -1.5, -195.8748
        self.Li6He3aap_T9, self.Li6He3aap_median, self.Li6He3aap_expsigma = np.loadtxt(oth + "Li6He3aap.txt", unpack=True)
        # Li6t -> aan
        self.alpha_Li6taan, self.beta_Li6taan, self.gamma_Li6taan = 7.2333e-10, -1.5, -187.0131
        self.Li6taan_T9, self.Li6taan_median, self.Li6taan_expsigma = np.loadtxt(oth + "Li6taan.txt", unpack=True)
        # Li6t -> Li8p
        self.alpha_Li6tLi8p, self.beta_Li6tLi8p, self.gamma_Li6tLi8p = 2.0167, 0., -9.306
        self.Li6tLi8p_T9, self.Li6tLi8p_median, self.Li6tLi8p_expsigma = np.loadtxt(oth + "Li6tLi8p.txt", unpack=True)
        # Li7He3 -> Li6a
        self.alpha_Li7He3Li6a, self.beta_Li7He3Li6a, self.gamma_Li7He3Li6a = 2.1972, 0., -154.6607
        self.Li7He3Li6a_T9, self.Li7He3Li6a_median, self.Li7He3Li6a_expsigma = np.loadtxt(oth + "Li7He3Li6a.txt", unpack=True)
        # Li8He3 -> Li7a
        self.alpha_Li8He3Li7a, self.beta_Li8He3Li7a, self.gamma_Li8He3Li7a = 1.9994, 0., -215.2055
        self.Li8He3Li7a_T9, self.Li8He3Li7a_median, self.Li8He3Li7a_expsigma = np.loadtxt(oth + "Li8He3Li7a.txt", unpack=True)
        # Be7t -> Li6a
        self.alpha_Be7tLi6a, self.beta_Be7tLi6a, self.gamma_Be7tLi6a = 2.1977, 0., -164.8783
        self.Be7tLi6a_T9, self.Be7tLi6a_median, self.Be7tLi6a_expsigma = np.loadtxt(oth + "Be7tLi6a.txt", unpack=True)
        # B8t -> Be7a
        self.alpha_B8tBe7a, self.beta_B8tBe7a, self.gamma_B8tBe7a = 1.9999, 0., -228.3344
        self.B8tBe7a_T9, self.B8tBe7a_median, self.B8tBe7a_expsigma = np.loadtxt(oth + "B8tBe7a.txt", unpack=True)
        # B8n -> Li6He3
        self.alpha_B8nLi6He3, self.beta_B8nLi6He3, self.gamma_B8nLi6He3 = 0.49669, 0., -22.9167
        self.B8nLi6He3_T9, self.B8nLi6He3_median, self.B8nLi6He3_expsigma = np.loadtxt(oth + "B8nLi6He3.txt", unpack=True)
        # B8n -> Be7d
        self.alpha_B8nBe7d, self.beta_B8nBe7d, self.gamma_B8nBe7d = 0.36119, 0., -24.2194
        self.B8nBe7d_T9, self.B8nBe7d_median, self.B8nBe7d_expsigma = np.loadtxt(oth + "B8nBe7d.txt", unpack=True)
        # Li6t -> Li7d
        self.alpha_Li6tLi7d, self.beta_Li6tLi7d, self.gamma_Li6tLi7d = 0.72734, 0., -11.5332
        self.Li6tLi7d_T9, self.Li6tLi7d_median, self.Li6tLi7d_expsigma = np.loadtxt(oth + "Li6tLi7d.txt", unpack=True)
        # Li6He3 -> Be7d
        self.alpha_Li6He3Be7d, self.beta_Li6He3Be7d, self.gamma_Li6He3Be7d = 0.72719, 0., -1.3157
        self.Li6He3Be7d_T9, self.Li6He3Be7d_median, self.Li6He3Be7d_expsigma = np.loadtxt(oth + "Li6He3Be7d.txt", unpack=True)
        # Li7He3 -> aad
        self.alpha_Li7He3aad, self.beta_Li7He3aad, self.gamma_Li7He3aad = 2.8700e-10, -1.5, -137.5575
        self.Li7He3aad_T9, self.Li7He3aad_median, self.Li7He3aad_expsigma = np.loadtxt(oth + "Li7He3aad.txt", unpack=True)
        # Li8He3 -> aat
        self.alpha_Li8He3aat, self.beta_Li8He3aat, self.gamma_Li8He3aat = 3.5907e-10, -1.5, -186.5821
        self.Li8He3aat_T9, self.Li8He3aat_median, self.Li8He3aat_expsigma = np.loadtxt(oth + "Li8He3aat.txt", unpack=True)
        # Be7t -> aad
        self.alpha_Be7taad, self.beta_Be7taad, self.gamma_Be7taad = 2.8706e-10, -1.5, -147.7751
        self.Be7taad_T9, self.Be7taad_median, self.Be7taad_expsigma = np.loadtxt(oth + "Be7taad.txt", unpack=True)
        # Be7t -> Li7He3
        self.alpha_Be7tLi7He3, self.beta_Be7tLi7He3, self.gamma_Be7tLi7He3 = 1.0002, 0., -10.2176
        self.Be7tLi7He3_T9, self.Be7tLi7He3_median, self.Be7tLi7He3_expsigma = np.loadtxt(oth + "Be7tLi7He3.txt", unpack=True)
        # B8d -> Be7He3
        self.alpha_B8dBe7He3, self.beta_B8dBe7He3, self.gamma_B8dBe7He3 = 1.2514, 0., -62.1535
        self.B8dBe7He3_T9, self.B8dBe7He3_median, self.B8dBe7He3_expsigma = np.loadtxt(oth + "B8dBe7He3.txt", unpack=True)
        # B8t -> aaHe3
        self.alpha_B8taaHe3, self.beta_B8taaHe3, self.gamma_B8taaHe3 = 3.5922e-10, -1.5, -209.9285
        self.B8taaHe3_T9, self.B8taaHe3_median, self.B8taaHe3_expsigma = np.loadtxt(oth + "B8taaHe3.txt", unpack=True)
        # Be7He3p -> paa
        self.alpha_Be7He3ppaa, self.beta_Be7He3ppaa, self.gamma_Be7He3ppaa = 1.2201e-19, -3., -130.8113
        self.Be7He3ppaa_T9, self.Be7He3ppaa_median, self.Be7He3ppaa_expsigma = np.loadtxt(oth + "Be7He3ppaa.txt", unpack=True)
        # dd -> ag
        self.alpha_ddag, self.beta_ddag, self.gamma_ddag = 4.5310e+10, 1.5, -276.7271
        self.ddag_T9, self.ddag_median, self.ddag_expsigma = np.loadtxt(oth + "ddag.txt", unpack=True)
        # He3He3 -> app
        self.alpha_He3He3app, self.beta_He3He3app, self.gamma_He3He3app = 3.3915e-10, -1.5, -149.2290
        self.He3He3app_T9, self.He3He3app_median, self.He3He3app_expsigma = np.loadtxt(oth + "He3He3app.txt", unpack=True)
        # Be7p -> B8g
        self.alpha_Be7pB8g, self.beta_Be7pB8g, self.gamma_Be7pB8g = 1.3063e+10, 1.5, -1.5825
        self.Be7pB8g_T9, self.Be7pB8g_median, self.Be7pB8g_expsigma = np.loadtxt(oth + "Be7pB8g.txt", unpack=True)
        # Li7d -> aan
        self.alpha_Li7daan, self.beta_Li7daan, self.gamma_Li7daan = 9.9435e-10, -1.5, -175.4916
        self.Li7daan_T9, self.Li7daan_median, self.Li7daan_expsigma = np.loadtxt(oth + "Li7daan.txt", unpack=True)
        # dn -> tg
        self.alpha_dntg, self.beta_dntg, self.gamma_dntg = 1.6364262e+10, 1.5, -72.612132
        self.dntg_T9, self.dntg_median, self.dntg_expsigma = np.loadtxt(oth + "dntg.txt", unpack=True)
        # tt -> ann
        self.alpha_ttann, self.beta_ttann, self.gamma_ttann = 3.3826187e-10, -1.5, -131.50322
        self.ttann_T9, self.ttann_median, self.ttann_expsigma = np.loadtxt(oth + "ttann.txt", unpack=True)
        # He3n -> ag
        self.alpha_He3nag, self.beta_He3nag, self.gamma_He3nag = 2.6152351e+10, 1.5, -238.79338
        self.He3nag_T9, self.He3nag_median, self.He3nag_expsigma = np.loadtxt(oth + "He3nag.txt", unpack=True)
        # He3t -> ad
        self.alpha_He3tad, self.beta_He3tad, self.gamma_He3tad = 1.5981381, 0., -166.18124
        self.He3tad_T9, self.He3tad_median, self.He3tad_expsigma = np.loadtxt(oth + "He3tad.txt", unpack=True)
        # He3t -> anp
        self.alpha_He3tanp, self.beta_He3tanp, self.gamma_He3tanp = 3.3886566e-10, -1.5, -140.36623
        self.He3tanp_T9, self.He3tanp_median, self.He3tanp_expsigma = np.loadtxt(oth + "He3tanp.txt", unpack=True)
        # Li7t -> aan
        self.alpha_Li7taan, self.beta_Li7taan, self.gamma_Li7taan = 1.2153497e-19, -3., -102.86767
        self.Li7taan_T9, self.Li7taan_median, self.Li7taan_expsigma = np.loadtxt(oth + "Li7taan.txt", unpack=True)
        # Li7He3 -> aanp
        self.alpha_Li7He3aanp, self.beta_Li7He3aanp, self.gamma_Li7He3aanp = 6.0875952e-20, -3., -111.73068
        self.Li7He3aanp_T9, self.Li7He3aanp_median, self.Li7He3aanp_expsigma = np.loadtxt(oth + "Li7He3aanp.txt", unpack=True)
        # Li8d -> Li7t
        self.alpha_Li8dLi7t, self.beta_Li8dLi7t, self.gamma_Li8dLi7t = 1.2509926, 0., -49.02453
        self.Li8dLi7t_T9, self.Li8dLi7t_median, self.Li8dLi7t_expsigma = np.loadtxt(oth + "Li8dLi7t.txt", unpack=True)
        # Be7t -> aanp
        self.alpha_Be7taanp, self.beta_Be7taanp, self.gamma_Be7taanp = 6.0898077e-20, -3., -121.9483
        self.Be7taanp_T9, self.Be7taanp_median, self.Be7taanp_expsigma = np.loadtxt(oth + "Be7taanp.txt", unpack=True)
        # Be7He3 -> aapp
        self.alpha_Be7He3aapp, self.beta_Be7He3aapp, self.gamma_Be7He3aapp = 1.2201356e-19, -3., -130.81131
        self.Be7He3aapp_T9, self.Be7He3aapp_median, self.Be7He3aapp_expsigma = np.loadtxt(oth + "Be7He3aapp.txt", unpack=True)
        # Li6n -> ta
        self.alpha_Li6nta, self.beta_Li6nta, self.gamma_Li6nta = 1.0691921, 0., -55.509875
        self.Li6nta_T9, self.Li6nta_median, self.Li6nta_expsigma = np.loadtxt(oth + "Li6nta.txt", unpack=True)
        # He3t -> Li6g
        self.alpha_He3tLi6g, self.beta_He3tLi6g, self.gamma_He3tLi6g = 2.4459918e+10, 1.5, -183.2835
        self.He3tLi6g_T9, self.He3tLi6g_median, self.He3tLi6g_expsigma = np.loadtxt(oth + "He3tLi6g.txt", unpack=True)
        # an -> pLi6g
        self.alpha_anpLi6g, self.beta_anpLi6g, self.gamma_anpLi6g = 7.2181753e+19, 3., -42.917276
        self.anpLi6g_T9, self.anpLi6g_median, self.anpLi6g_expsigma = np.loadtxt(oth + "anpLi6g.txt", unpack=True)
        # Li6n -> Li7g
        self.alpha_Li6nLi7g, self.beta_Li6nLi7g, self.gamma_Li6nLi7g = 1.1903305e+10, 1.5, -84.145424
        self.Li6nLi7g_T9, self.Li6nLi7g_median, self.Li6nLi7g_expsigma = np.loadtxt(oth + "Li6nLi7g.txt", unpack=True)
        # Li6d -> Li7p
        self.alpha_Li6dLi7p, self.beta_Li6dLi7p, self.gamma_Li6dLi7p = 2.5239503, 0., -58.330405
        self.Li6dLi7p_T9, self.Li6dLi7p_median, self.Li6dLi7p_expsigma = np.loadtxt(oth + "Li6dLi7p.txt", unpack=True)
        # Li6d -> Be7n
        self.alpha_Li6dBe7n, self.beta_Li6dBe7n, self.gamma_Li6dBe7n = 2.5185377, 0., -39.249773
        self.Li6dBe7n_T9, self.Li6dBe7n_median, self.Li6dBe7n_expsigma = np.loadtxt(oth + "Li6dBe7n.txt", unpack=True)
        # Li7n -> Li8g
        self.alpha_Li7nLi8g, self.beta_Li7nLi8g, self.gamma_Li7nLi8g = 1.3081022e+10, 1.5, -23.587602
        self.Li7nLi8g_T9, self.Li7nLi8g_median, self.Li7nLi8g_expsigma = np.loadtxt(oth + "Li7nLi8g.txt", unpack=True)
        # Li7d -> Li8p
        self.alpha_Li7dLi8p, self.beta_Li7dLi8p, self.gamma_Li7dLi8p = 2.7736709, 0., 2.2274166
        self.Li7dLi8p_T9, self.Li7dLi8p_median, self.Li7dLi8p_expsigma = np.loadtxt(oth + "Li7dLi8p.txt", unpack=True)
        # Li8p -> aan
        self.alpha_Li8paan, self.beta_Li8paan, self.gamma_Li8paan = 3.5851946e-10, -1.5, -177.70722
        self.Li8paan_T9, self.Li8paan_median, self.Li8paan_expsigma = np.loadtxt(oth + "Li8paan.txt", unpack=True)
        # an -> nHe6g
        self.alpha_annHe6g, self.beta_annHe6g, self.gamma_annHe6g = 1.0837999e+20, 3., -11.319626
        self.annHe6g_T9, self.annHe6g_median, self.annHe6g_expsigma = np.loadtxt(oth + "annHe6g.txt", unpack=True)
        # pp -> ndp
        self.alpha_ppndp, self.beta_ppndp, self.gamma_ppndp = 2.3580703e+9, 1.5, -25.815019
        self.ppndp_T9, self.ppndp_median, self.ppndp_expsigma = np.loadtxt(oth + "ppndp.txt", unpack=True)
        # Li7t -> aann
        self.alpha_Li7taann, self.beta_Li7taann, self.gamma_Li7taann = 1.2153497e-19, -3., -102.86767
        self.Li7taann_T9, self.Li7taann_median, self.Li7taann_expsigma = np.loadtxt(oth + "Li7taann.txt", unpack=True)

        if cfg.verbose_flag:
            print("[rates] All extra 51 rate tables loaded.")