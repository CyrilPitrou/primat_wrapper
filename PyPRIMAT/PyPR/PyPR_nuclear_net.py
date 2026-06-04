# -*- coding: utf-8 -*-
"""
PyPR_nuclear_net.py
===================
Unified nuclear network for BBN (12-reaction or 63-reaction).

``UpdateNuclearRates`` receives a ``NuclearData`` instance (for rate tables)
and a ``PyPRConfig`` instance (for detailed-balance coefficients and NP flags).
When ``cfg.smallnet_flag`` is True, only the 12 key reactions are loaded.
When False, all 63 reactions are loaded.
"""

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Reaction lists
# ---------------------------------------------------------------------------

_KEY12_REACTIONS = [
    'npdg', 'dpHe3g', 'ddHe3n', 'ddtp', 'tpag', 'tdan', 'taLi7g',
    'He3ntp', 'He3dap', 'He3aBe7g', 'Be7nLi7p', 'Li7paa',
]

_EXTRA29_LINEAR = [
    'Li7paag', 'Be7naa', 'Be7daap', 'daLi6g', 'Li6pBe7g', 'Li6pHe3a',
    'B8naap', 'Li6He3aap', 'Li6taan', 'Li6tLi8p', 'Li7He3Li6a', 'Li8He3Li7a',
    'Be7tLi6a', 'B8tBe7a', 'B8nLi6He3', 'B8nBe7d', 'Li6tLi7d', 'Li6He3Be7d',
    'Li7He3aad', 'Li8He3aat', 'Be7taad', 'Be7tLi7He3', 'B8dBe7He3', 'B8taaHe3',
    'Be7He3ppaa', 'ddag', 'He3He3app', 'Be7pB8g', 'Li7daan',
]

_EXTRA22_QUADRATIC = [
    'dntg', 'ttann', 'He3nag', 'He3tad', 'He3tanp', 'Li7taan', 'Li7He3aanp',
    'Li8dLi7t', 'Be7taanp', 'Be7He3aapp', 'Li6nta', 'He3tLi6g', 'anpLi6g',
    'Li6nLi7g', 'Li6dLi7p', 'Li6dBe7n', 'Li7nLi8g', 'Li7dLi8p', 'Li8paan',
    'annHe6g', 'ppndp', 'Li7taann',
]

_ALL_REACTIONS = _KEY12_REACTIONS + _EXTRA29_LINEAR + _EXTRA22_QUADRATIC


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Arithmetic kernels for rhs / rhsMT / rhsLT — JIT-compiled when numba is
# available.  Signature: (Y_array, rhoBBN, r_rates_array) -> tuple.
# Call _setup_nuclear_rhs_impls(cfg.numba_flag) to initialise.
# ---------------------------------------------------------------------------

_rhs_impl = None
_rhsMT_impl = None
_rhsLT_impl = None


def _rhs_arith(Y, rhoBBN, r):
    Yn = Y[0]
    Yp = Y[1]
    Yd = Y[2]
    Yt = Y[3]
    YHe3 = Y[4]
    Ya = Y[5]
    YLi7 = Y[6]
    YBe7 = Y[7]
    nTOp_f = r[0]; nTOp_b = r[1]
    npdg_f = r[2]; npdg_b = r[3]
    dpHe3g_f = r[4]; dpHe3g_b = r[5]
    ddHe3n_f = r[6]; ddHe3n_b = r[7]
    ddtp_f = r[8]; ddtp_b = r[9]
    tpag_f = r[10]; tpag_b = r[11]
    tdan_f = r[12]; tdan_b = r[13]
    taLi7g_f = r[14]; taLi7g_b = r[15]
    He3ntp_f = r[16]; He3ntp_b = r[17]
    He3dap_f = r[18]; He3dap_b = r[19]
    He3aBe7g_f = r[20]; He3aBe7g_b = r[21]
    Be7nLi7p_f = r[22]; Be7nLi7p_b = r[23]
    Li7paa_f = r[24]; Li7paa_b = r[25]
    dYn = (-nTOp_f*Yn + nTOp_b*Yp
            - rhoBBN*npdg_f*Yn*Yp + npdg_b*Yd
            + 0.5*rhoBBN*ddHe3n_f*Yd*Yd
            + rhoBBN*He3ntp_b*Yp*Yt
            + rhoBBN*tdan_f*Yd*Yt
            - rhoBBN*He3ntp_f*Yn*YHe3
            - rhoBBN*ddHe3n_b*Yn*YHe3
            - rhoBBN*tdan_b*Yn*Ya
            + rhoBBN*Be7nLi7p_b*Yp*YLi7
            - rhoBBN*Be7nLi7p_f*Yn*YBe7)
    dYp = (nTOp_f*Yn - nTOp_b*Yp
            - rhoBBN*npdg_f*Yn*Yp + npdg_b*Yd
            - rhoBBN*dpHe3g_f*Yp*Yd
            + 0.5*rhoBBN*ddtp_f*Yd*Yd
            - rhoBBN*tpag_f*Yp*Yt
            - rhoBBN*ddtp_b*Yp*Yt
            - rhoBBN*He3ntp_b*Yp*Yt
            + dpHe3g_b*YHe3
            + rhoBBN*He3ntp_f*Yn*YHe3
            + rhoBBN*He3dap_f*Yd*YHe3
            + tpag_b*Ya
            - rhoBBN*He3dap_b*Yp*Ya
            + 0.5*rhoBBN*Li7paa_b*Ya*Ya
            - rhoBBN*Li7paa_f*Yp*YLi7
            - rhoBBN*Be7nLi7p_b*Yp*YLi7
            + rhoBBN*Be7nLi7p_f*Yn*YBe7)
    dYd = (rhoBBN*npdg_f*Yn*Yp - npdg_b*Yd
            - rhoBBN*dpHe3g_f*Yp*Yd
            - rhoBBN*ddHe3n_f*Yd*Yd
            - rhoBBN*ddtp_f*Yd*Yd
            + 2.*rhoBBN*ddtp_b*Yp*Yt
            - rhoBBN*tdan_f*Yd*Yt
            + dpHe3g_b*YHe3
            + 2.*rhoBBN*ddHe3n_b*Yn*YHe3
            - rhoBBN*He3dap_f*Yd*YHe3
            + rhoBBN*tdan_b*Yn*Ya
            + rhoBBN*He3dap_b*Yp*Ya)
    dYt = (0.5*rhoBBN*ddtp_f*Yd*Yd
            - rhoBBN*tpag_f*Yp*Yt
            - rhoBBN*ddtp_b*Yp*Yt
            - rhoBBN*He3ntp_b*Yp*Yt
            - rhoBBN*tdan_f*Yd*Yt
            + rhoBBN*He3ntp_f*Yn*YHe3
            + tpag_b*Ya
            + rhoBBN*tdan_b*Yn*Ya
            - rhoBBN*taLi7g_f*Yt*Ya
            + taLi7g_b*YLi7)
    dYHe3 = (rhoBBN*dpHe3g_f*Yp*Yd
            + 0.5*rhoBBN*ddHe3n_f*Yd*Yd
            + rhoBBN*He3ntp_b*Yp*Yt
            - dpHe3g_b*YHe3
            - rhoBBN*He3ntp_f*Yn*YHe3
            - rhoBBN*ddHe3n_b*Yn*YHe3
            - rhoBBN*He3dap_f*Yd*YHe3
            + rhoBBN*He3dap_b*Yp*Ya
            - rhoBBN*He3aBe7g_f*YHe3*Ya
            + He3aBe7g_b*YBe7)
    dYa = (rhoBBN*tpag_f*Yp*Yt
            + rhoBBN*tdan_f*Yd*Yt
            + rhoBBN*He3dap_f*Yd*YHe3
            - tpag_b*Ya
            - rhoBBN*tdan_b*Yn*Ya
            - rhoBBN*He3dap_b*Yp*Ya
            - rhoBBN*taLi7g_f*Yt*Ya
            - rhoBBN*He3aBe7g_f*YHe3*Ya
            - rhoBBN*Li7paa_b*Ya*Ya
            + taLi7g_b*YLi7
            + 2*rhoBBN*Li7paa_f*Yp*YLi7
            + He3aBe7g_b*YBe7)
    dYLi7 = (rhoBBN*taLi7g_f*Yt*Ya
            + 0.5*rhoBBN*Li7paa_b*Ya*Ya
            - taLi7g_b*YLi7
            - rhoBBN*Li7paa_f*Yp*YLi7
            - rhoBBN*Be7nLi7p_b*Yp*YLi7
            + rhoBBN*Be7nLi7p_f*Yn*YBe7)
    dYBe7 = (rhoBBN*He3aBe7g_f*YHe3*Ya
            + rhoBBN*Be7nLi7p_b*Yp*YLi7
            - He3aBe7g_b*YBe7
            - rhoBBN*Be7nLi7p_f*Yn*YBe7)
    return (dYn, dYp, dYd, dYt, dYHe3, dYa, dYLi7, dYBe7)


def _rhsMT_arith(Y, rhoBBN, r):
    Yn1p0 = Y[0]
    Yn0p1 = Y[1]
    Yn1p1 = Y[2]
    Yn2p1 = Y[3]
    Yn1p2 = Y[4]
    Yn2p2 = Y[5]
    Yn4p3 = Y[6]
    Yn3p4 = Y[7]
    Yn4p2 = Y[8]
    Yn5p3 = Y[9]
    Yn3p3 = Y[10]
    Yn3p5 = Y[11]
    nTOp_f = r[0]; nTOp_b = r[1]
    Be7daap_f = r[2]; Be7daap_b = r[3]
    Be7nLi7p_f = r[4]; Be7nLi7p_b = r[5]
    Be7naa_f = r[6]; Be7naa_b = r[7]
    He3aBe7g_f = r[8]; He3aBe7g_b = r[9]
    He3dap_f = r[10]; He3dap_b = r[11]
    He3ntp_f = r[12]; He3ntp_b = r[13]
    Li6pBe7g_f = r[14]; Li6pBe7g_b = r[15]
    Li7paa_f = r[16]; Li7paa_b = r[17]
    Li7paag_f = r[18]; Li7paag_b = r[19]
    daLi6g_f = r[20]; daLi6g_b = r[21]
    ddHe3n_f = r[22]; ddHe3n_b = r[23]
    ddtp_f = r[24]; ddtp_b = r[25]
    dpHe3g_f = r[26]; dpHe3g_b = r[27]
    npdg_f = r[28]; npdg_b = r[29]
    taLi7g_f = r[30]; taLi7g_b = r[31]
    tdan_f = r[32]; tdan_b = r[33]
    tpag_f = r[34]; tpag_b = r[35]
    dYn = -nTOp_f*Yn1p0 + nTOp_b*Yn0p1 + rhoBBN*(0.5*ddHe3n_f*Yn1p1*Yn1p1 - npdg_f*Yn0p1*Yn1p0 + He3ntp_b*Yn0p1*Yn2p1 + tdan_f*Yn1p1*Yn2p1 - (He3ntp_f + ddHe3n_b)*Yn1p0*Yn1p2 - tdan_b*Yn1p0*Yn2p2 + Be7nLi7p_b*Yn0p1*Yn4p3 - Be7nLi7p_f*Yn1p0*Yn3p4) + npdg_b*Yn1p1 + rhoBBN*(-Be7naa_f*Yn1p0*Yn3p4) + rhoBBN*(0.5*Be7naa_b*Yn2p2*Yn2p2)
    dYp = nTOp_f*Yn1p0 - nTOp_b*Yn0p1 + rhoBBN*(0.5*ddtp_f*Yn1p1*Yn1p1 - npdg_f*Yn0p1*Yn1p0 - dpHe3g_f*Yn0p1*Yn1p1 - (tpag_f + ddtp_b + He3ntp_b)*Yn0p1*Yn2p1 + He3ntp_f*Yn1p0*Yn1p2 + He3dap_f*Yn1p1*Yn1p2 - He3dap_b*Yn0p1*Yn2p2 + 0.5*Li7paa_b*Yn2p2*Yn2p2 - (Li7paa_f + Be7nLi7p_b)*Yn0p1*Yn4p3 + Be7nLi7p_f*Yn1p0*Yn3p4) + npdg_b*Yn1p1 + dpHe3g_b*Yn1p2 + tpag_b*Yn2p2 + rhoBBN*(- 0.5*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*Yn2p2 + Be7daap_f*Yn1p1*Yn3p4) + rhoBBN*(-Li6pBe7g_f*Yn0p1*Yn3p3) + Li6pBe7g_b*Yn3p4 + rhoBBN*(-Li7paag_f*Yn0p1*Yn4p3) + 0.5*rhoBBN*Li7paag_b*Yn2p2*Yn2p2
    dYd = rhoBBN*(npdg_f*Yn0p1*Yn1p0 - dpHe3g_f*Yn0p1*Yn1p1 - (ddHe3n_f + ddtp_f)*Yn1p1*Yn1p1 + 2.*ddtp_b*Yn0p1*Yn2p1 - tdan_f*Yn1p1*Yn2p1 + 2.*ddHe3n_b*Yn1p0*Yn1p2 - He3dap_f*Yn1p1*Yn1p2 + tdan_b*Yn1p0*Yn2p2 + He3dap_b*Yn0p1*Yn2p2) - npdg_b*Yn1p1 + dpHe3g_b*Yn1p2 + rhoBBN*(-Be7daap_f*Yn1p1*Yn3p4 + 0.5*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*Yn2p2) + rhoBBN*(-daLi6g_f*Yn1p1*Yn2p2) + daLi6g_b*Yn3p3
    dYt = rhoBBN*(0.5*ddtp_f*Yn1p1*Yn1p1 - (tpag_f+ddtp_b+He3ntp_b)*Yn0p1*Yn2p1 - tdan_f*Yn1p1*Yn2p1 + He3ntp_f*Yn1p0*Yn1p2 + tdan_b*Yn1p0*Yn2p2 - taLi7g_f*Yn2p1*Yn2p2) + tpag_b*Yn2p2 + taLi7g_b*Yn4p3
    dYHe3 = rhoBBN*(dpHe3g_f*Yn0p1*Yn1p1 + 0.5*ddHe3n_f*Yn1p1*Yn1p1 + He3ntp_b*Yn0p1*Yn2p1 - He3dap_f*Yn1p1*Yn1p2 + He3dap_b*Yn0p1*Yn2p2 - (He3ntp_f+ddHe3n_b)*Yn1p0*Yn1p2 - He3aBe7g_f*Yn1p2*Yn2p2) + He3aBe7g_b*Yn3p4 - dpHe3g_b*Yn1p2
    dYa = rhoBBN*(tpag_f*Yn0p1*Yn2p1 + tdan_f*Yn1p1*Yn2p1 + He3dap_f*Yn1p1*Yn1p2 - tdan_b*Yn1p0*Yn2p2 - He3dap_b*Yn0p1*Yn2p2 - taLi7g_f*Yn2p1*Yn2p2 - He3aBe7g_f*Yn1p2*Yn2p2 - Li7paa_b*Yn2p2*Yn2p2 + 2.*Li7paa_f*Yn0p1*Yn4p3) + He3aBe7g_b*Yn3p4 - tpag_b*Yn2p2 + taLi7g_b*Yn4p3 + rhoBBN*(-Be7naa_b*Yn2p2*Yn2p2) + rhoBBN*(2*Be7naa_f*Yn1p0*Yn3p4) + rhoBBN*(- rhoBBN*Be7daap_b*Yn2p2*Yn2p2*Yn0p1 + 2*Be7daap_f*Yn1p1*Yn3p4) + rhoBBN*(-daLi6g_f*Yn2p2*Yn1p1) + daLi6g_b*Yn3p3 + (-rhoBBN*Li7paag_b*Yn2p2*Yn2p2) + rhoBBN*(2*Li7paag_f*Yn0p1*Yn4p3)
    dYLi7 = rhoBBN*(taLi7g_f*Yn2p1*Yn2p2 + 0.5*Li7paa_b*Yn2p2*Yn2p2 - (Li7paa_f+Be7nLi7p_b)*Yn0p1*Yn4p3 + Be7nLi7p_f*Yn1p0*Yn3p4) - taLi7g_b*Yn4p3 + rhoBBN*(-Li7paag_f*Yn4p3*Yn0p1) + 0.5*rhoBBN*Li7paag_b*Yn2p2*Yn2p2
    dYBe7 = rhoBBN*(He3aBe7g_f*Yn1p2*Yn2p2 + Be7nLi7p_b*Yn0p1*Yn4p3 - Be7nLi7p_f*Yn1p0*Yn3p4) - He3aBe7g_b*Yn3p4 + rhoBBN*(-Be7naa_f*Yn3p4*Yn1p0) + rhoBBN*(0.5*Be7naa_b*Yn2p2*Yn2p2) + rhoBBN*(-Be7daap_f*Yn3p4*Yn1p1 + 0.5*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*Yn2p2) + (-Li6pBe7g_b*Yn3p4) + rhoBBN*(Li6pBe7g_f*Yn0p1*Yn3p3)
    dYHe6 = 0.
    dYLi8 = 0.
    dYLi6 = (-daLi6g_b*Yn3p3) + rhoBBN*(daLi6g_f*Yn1p1*Yn2p2) + rhoBBN*(-Li6pBe7g_f*Yn3p3*Yn0p1) + (Li6pBe7g_b*Yn3p4)
    dYB8 = 0.
    return (dYn, dYp, dYd, dYt, dYHe3, dYa, dYLi7, dYBe7, dYHe6, dYLi8, dYLi6, dYB8)


def _rhsLT_arith(Y, rhoBBN, r):
    Yn1p0 = Y[0]
    Yn0p1 = Y[1]
    Yn1p1 = Y[2]
    Yn2p1 = Y[3]
    Yn1p2 = Y[4]
    Yn2p2 = Y[5]
    Yn4p3 = Y[6]
    Yn3p4 = Y[7]
    Yn4p2 = Y[8]
    Yn5p3 = Y[9]
    Yn3p3 = Y[10]
    Yn3p5 = Y[11]
    nTOp_f = r[0]; nTOp_b = r[1]
    B8dBe7He3_f = r[2]; B8dBe7He3_b = r[3]
    B8nBe7d_f = r[4]; B8nBe7d_b = r[5]
    B8nLi6He3_f = r[6]; B8nLi6He3_b = r[7]
    B8naap_f = r[8]; B8naap_b = r[9]
    B8tBe7a_f = r[10]; B8tBe7a_b = r[11]
    B8taaHe3_f = r[12]; B8taaHe3_b = r[13]
    Be7He3aapp_f = r[14]; Be7He3aapp_b = r[15]
    Be7He3ppaa_f = r[16]; Be7He3ppaa_b = r[17]
    Be7daap_f = r[18]; Be7daap_b = r[19]
    Be7nLi7p_f = r[20]; Be7nLi7p_b = r[21]
    Be7naa_f = r[22]; Be7naa_b = r[23]
    Be7pB8g_f = r[24]; Be7pB8g_b = r[25]
    Be7tLi6a_f = r[26]; Be7tLi6a_b = r[27]
    Be7tLi7He3_f = r[28]; Be7tLi7He3_b = r[29]
    Be7taad_f = r[30]; Be7taad_b = r[31]
    Be7taanp_f = r[32]; Be7taanp_b = r[33]
    He3He3app_f = r[34]; He3He3app_b = r[35]
    He3aBe7g_f = r[36]; He3aBe7g_b = r[37]
    He3dap_f = r[38]; He3dap_b = r[39]
    He3nag_f = r[40]; He3nag_b = r[41]
    He3ntp_f = r[42]; He3ntp_b = r[43]
    He3tLi6g_f = r[44]; He3tLi6g_b = r[45]
    He3tad_f = r[46]; He3tad_b = r[47]
    He3tanp_f = r[48]; He3tanp_b = r[49]
    Li6He3Be7d_f = r[50]; Li6He3Be7d_b = r[51]
    Li6He3aap_f = r[52]; Li6He3aap_b = r[53]
    Li6dBe7n_f = r[54]; Li6dBe7n_b = r[55]
    Li6dLi7p_f = r[56]; Li6dLi7p_b = r[57]
    Li6nLi7g_f = r[58]; Li6nLi7g_b = r[59]
    Li6nta_f = r[60]; Li6nta_b = r[61]
    Li6pBe7g_f = r[62]; Li6pBe7g_b = r[63]
    Li6pHe3a_f = r[64]; Li6pHe3a_b = r[65]
    Li6tLi7d_f = r[66]; Li6tLi7d_b = r[67]
    Li6tLi8p_f = r[68]; Li6tLi8p_b = r[69]
    Li6taan_f = r[70]; Li6taan_b = r[71]
    Li7He3Li6a_f = r[72]; Li7He3Li6a_b = r[73]
    Li7He3aad_f = r[74]; Li7He3aad_b = r[75]
    Li7He3aanp_f = r[76]; Li7He3aanp_b = r[77]
    Li7dLi8p_f = r[78]; Li7dLi8p_b = r[79]
    Li7daan_f = r[80]; Li7daan_b = r[81]
    Li7nLi8g_f = r[82]; Li7nLi8g_b = r[83]
    Li7paa_f = r[84]; Li7paa_b = r[85]
    Li7paag_f = r[86]; Li7paag_b = r[87]
    Li7taann_f = r[88]; Li7taann_b = r[89]
    Li8He3Li7a_f = r[90]; Li8He3Li7a_b = r[91]
    Li8He3aat_f = r[92]; Li8He3aat_b = r[93]
    Li8dLi7t_f = r[94]; Li8dLi7t_b = r[95]
    Li8paan_f = r[96]; Li8paan_b = r[97]
    annHe6g_f = r[98]; annHe6g_b = r[99]
    anpLi6g_f = r[100]; anpLi6g_b = r[101]
    daLi6g_f = r[102]; daLi6g_b = r[103]
    ddHe3n_f = r[104]; ddHe3n_b = r[105]
    ddag_f = r[106]; ddag_b = r[107]
    ddtp_f = r[108]; ddtp_b = r[109]
    dntg_f = r[110]; dntg_b = r[111]
    dpHe3g_f = r[112]; dpHe3g_b = r[113]
    npdg_f = r[114]; npdg_b = r[115]
    ppndp_f = r[116]; ppndp_b = r[117]
    taLi7g_f = r[118]; taLi7g_b = r[119]
    tdan_f = r[120]; tdan_b = r[121]
    tpag_f = r[122]; tpag_b = r[123]
    ttann_f = r[124]; ttann_b = r[125]
    dYn = -nTOp_f*Yn1p0 + nTOp_b*Yn0p1 - rhoBBN*npdg_f*Yn1p0*Yn0p1 - 0.5*rhoBBN*rhoBBN* ppndp_f*Yn1p0*Yn0p1*Yn0p1 + npdg_b*Yn1p1 - rhoBBN*dntg_f*Yn1p0*Yn1p1 + rhoBBN*ppndp_b*Yn0p1*Yn1p1 + 0.5*rhoBBN*ddHe3n_f*Yn1p1*Yn1p1 + dntg_b*Yn2p1 + rhoBBN*He3ntp_b*Yn0p1*Yn2p1 + rhoBBN*tdan_f*Yn1p1*Yn2p1 + rhoBBN*ttann_f*Yn2p1*Yn2p1 - rhoBBN*He3ntp_f*Yn1p0*Yn1p2 - rhoBBN*He3nag_f*Yn1p0*Yn1p2 - rhoBBN*ddHe3n_b*Yn1p0*Yn1p2 + rhoBBN*He3tanp_f*Yn2p1*Yn1p2 + He3nag_b*Yn2p2 - rhoBBN*tdan_b*Yn1p0*Yn2p2 - rhoBBN*rhoBBN*annHe6g_f*Yn1p0*Yn1p0*Yn2p2 - rhoBBN*rhoBBN*ttann_b*Yn1p0*Yn1p0*Yn2p2 - rhoBBN*rhoBBN*anpLi6g_f*Yn1p0*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*He3tanp_b*Yn1p0*Yn0p1*Yn2p2 + rhoBBN*Li6nta_b*Yn2p1*Yn2p2 + 0.5*rhoBBN*Be7naa_b*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*Li6taan_b*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*Li7daan_b*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*Li8paan_b*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Li7taann_b*Yn1p0*Yn1p0*Yn2p2* Yn2p2 + 0.5*rhoBBN*rhoBBN*B8naap_b*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Li7He3aanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Be7taanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 2*annHe6g_b*Yn4p2 + anpLi6g_b*Yn3p3 - rhoBBN*Li6nta_f*Yn1p0*Yn3p3 - rhoBBN*Li6nLi7g_f*Yn1p0*Yn3p3 + rhoBBN*Li6dBe7n_f*Yn1p1*Yn3p3 + rhoBBN*Li6taan_f*Yn2p1*Yn3p3 + rhoBBN*B8nLi6He3_b*Yn1p2*Yn3p3 + Li6nLi7g_b*Yn4p3 - rhoBBN*Li7nLi8g_f*Yn1p0*Yn4p3 + rhoBBN*Be7nLi7p_b*Yn0p1*Yn4p3 + rhoBBN*Li7daan_f*Yn1p1*Yn4p3 + 2*rhoBBN*Li7taann_f*Yn2p1*Yn4p3 + rhoBBN*Li7He3aanp_f*Yn1p2*Yn4p3 + Li7nLi8g_b*Yn5p3 + rhoBBN*Li8paan_f*Yn0p1*Yn5p3 - rhoBBN*Be7nLi7p_f*Yn1p0*Yn3p4 - rhoBBN*Be7naa_f*Yn1p0*Yn3p4 - rhoBBN*Li6dBe7n_b*Yn1p0*Yn3p4 + rhoBBN*B8nBe7d_b*Yn1p1*Yn3p4 + rhoBBN*Be7taanp_f*Yn2p1*Yn3p4 - rhoBBN*B8naap_f*Yn1p0*Yn3p5 - rhoBBN*B8nLi6He3_f*Yn1p0*Yn3p5 - rhoBBN*B8nBe7d_f*Yn1p0*Yn3p5
    dYp = nTOp_f*Yn1p0 - nTOp_b*Yn0p1 - rhoBBN*npdg_f*Yn1p0*Yn0p1 - 0.5*rhoBBN*rhoBBN*ppndp_f*Yn1p0*Yn0p1*Yn0p1 + npdg_b*Yn1p1 - rhoBBN*dpHe3g_f*Yn0p1*Yn1p1 + rhoBBN*ppndp_b*Yn0p1*Yn1p1 + 0.5*rhoBBN*ddtp_f*Yn1p1*Yn1p1 - rhoBBN*tpag_f*Yn0p1*Yn2p1 - rhoBBN*ddtp_b*Yn0p1*Yn2p1 - rhoBBN*He3ntp_b*Yn0p1*Yn2p1 + dpHe3g_b*Yn1p2 + rhoBBN*He3ntp_f*Yn1p0*Yn1p2 + rhoBBN*He3dap_f*Yn1p1*Yn1p2 + rhoBBN*He3tanp_f*Yn2p1*Yn1p2 + rhoBBN*He3He3app_f*Yn1p2*Yn1p2 + tpag_b*Yn2p2 - rhoBBN*He3dap_b*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*anpLi6g_f*Yn1p0*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*He3tanp_b*Yn1p0*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*He3He3app_b*Yn0p1*Yn0p1*Yn2p2 + rhoBBN*Li6pHe3a_b*Yn1p2*Yn2p2 + 0.5*rhoBBN*Li7paa_b*Yn2p2*Yn2p2 + 0.5*rhoBBN*Li7paag_b*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li8paan_b*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*B8naap_b*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*Li6He3aap_b*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Li7He3aanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Be7taanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Be7He3ppaa_b*Yn0p1*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Be7He3aapp_b*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + anpLi6g_b*Yn3p3 - rhoBBN*Li6pBe7g_f*Yn0p1*Yn3p3 - rhoBBN*Li6pHe3a_f*Yn0p1*Yn3p3 + rhoBBN*Li6dLi7p_f*Yn1p1*Yn3p3 + rhoBBN*Li6tLi8p_f*Yn2p1*Yn3p3 + rhoBBN*Li6He3aap_f*Yn1p2*Yn3p3 - rhoBBN*Li7paa_f*Yn0p1*Yn4p3 - rhoBBN*Li7paag_f*Yn0p1*Yn4p3 - rhoBBN*Be7nLi7p_b*Yn0p1*Yn4p3 - rhoBBN*Li6dLi7p_b*Yn0p1*Yn4p3 + rhoBBN*Li7dLi8p_f*Yn1p1*Yn4p3 + rhoBBN*Li7He3aanp_f*Yn1p2*Yn4p3 - rhoBBN*Li8paan_f*Yn0p1*Yn5p3 - rhoBBN*Li6tLi8p_b*Yn0p1*Yn5p3 - rhoBBN*Li7dLi8p_b*Yn0p1*Yn5p3 + Li6pBe7g_b*Yn3p4 + rhoBBN*Be7nLi7p_f*Yn1p0*Yn3p4 - rhoBBN*Be7pB8g_f*Yn0p1*Yn3p4 + rhoBBN*Be7daap_f*Yn1p1*Yn3p4 + rhoBBN*Be7taanp_f*Yn2p1*Yn3p4 + 2.*rhoBBN*Be7He3ppaa_f*Yn1p2*Yn3p4 + 2.*rhoBBN*Be7He3aapp_f*Yn1p2*Yn3p4 + Be7pB8g_b*Yn3p5 + rhoBBN*B8naap_f*Yn1p0*Yn3p5
    dYd = rhoBBN*npdg_f*Yn1p0*Yn0p1 + 0.5*rhoBBN*rhoBBN*ppndp_f*Yn1p0*Yn0p1*Yn0p1 - npdg_b*Yn1p1 - rhoBBN*dntg_f*Yn1p0*Yn1p1 - rhoBBN*dpHe3g_f*Yn0p1*Yn1p1 - rhoBBN*ppndp_b*Yn0p1*Yn1p1 - rhoBBN*ddHe3n_f*Yn1p1*Yn1p1 - rhoBBN*ddtp_f*Yn1p1*Yn1p1 - rhoBBN*ddag_f*Yn1p1*Yn1p1 + dntg_b*Yn2p1 + 2*rhoBBN*ddtp_b*Yn0p1*Yn2p1 - rhoBBN*tdan_f*Yn1p1*Yn2p1 + dpHe3g_b*Yn1p2 + 2*rhoBBN*ddHe3n_b*Yn1p0*Yn1p2 - rhoBBN*He3dap_f*Yn1p1*Yn1p2 + rhoBBN*He3tad_f*Yn2p1*Yn1p2 + 2*ddag_b*Yn2p2 + rhoBBN*tdan_b*Yn1p0*Yn2p2 + rhoBBN*He3dap_b*Yn0p1*Yn2p2 - rhoBBN*daLi6g_f*Yn1p1*Yn2p2 - rhoBBN*He3tad_b*Yn1p1*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li7daan_b*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*Li7He3aad_b*Yn1p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*Be7taad_b*Yn1p1*Yn2p2*Yn2p2 + daLi6g_b*Yn3p3 - rhoBBN*Li6dLi7p_f*Yn1p1*Yn3p3 - rhoBBN*Li6dBe7n_f*Yn1p1*Yn3p3 + rhoBBN*Li6tLi7d_f*Yn2p1*Yn3p3 + rhoBBN*Li6He3Be7d_f*Yn1p2*Yn3p3 + rhoBBN*Li6dLi7p_b*Yn0p1*Yn4p3 - rhoBBN*Li7daan_f*Yn1p1*Yn4p3 - rhoBBN*Li7dLi8p_f*Yn1p1*Yn4p3 - rhoBBN*Li6tLi7d_b*Yn1p1*Yn4p3 + rhoBBN*Li8dLi7t_b*Yn2p1*Yn4p3 + rhoBBN*Li7He3aad_f*Yn1p2*Yn4p3 + rhoBBN*Li7dLi8p_b*Yn0p1*Yn5p3 - rhoBBN*Li8dLi7t_f*Yn1p1*Yn5p3 + rhoBBN*Li6dBe7n_b*Yn1p0*Yn3p4 - rhoBBN*Be7daap_f*Yn1p1*Yn3p4 - rhoBBN*B8nBe7d_b*Yn1p1*Yn3p4 - rhoBBN*Li6He3Be7d_b*Yn1p1*Yn3p4 + rhoBBN*Be7taad_f*Yn2p1*Yn3p4 + rhoBBN*B8dBe7He3_b*Yn1p2*Yn3p4 + rhoBBN*B8nBe7d_f*Yn1p0*Yn3p5 - rhoBBN*B8dBe7He3_f*Yn1p1*Yn3p5
    dYt = rhoBBN*dntg_f*Yn1p0*Yn1p1 + 0.5*rhoBBN*ddtp_f*Yn1p1*Yn1p1 - dntg_b*Yn2p1 - rhoBBN*tpag_f*Yn0p1*Yn2p1 - rhoBBN*ddtp_b*Yn0p1*Yn2p1 - rhoBBN*He3ntp_b*Yn0p1*Yn2p1 - rhoBBN*tdan_f*Yn1p1*Yn2p1 - rhoBBN*ttann_f*Yn2p1*Yn2p1 + rhoBBN*He3ntp_f*Yn1p0*Yn1p2 - rhoBBN*He3tad_f*Yn2p1*Yn1p2 - rhoBBN*He3tanp_f*Yn2p1*Yn1p2 - rhoBBN*He3tLi6g_f*Yn2p1*Yn1p2 + tpag_b*Yn2p2 + rhoBBN*tdan_b*Yn1p0*Yn2p2 + rhoBBN*rhoBBN*ttann_b*Yn1p0*Yn1p0*Yn2p2 + rhoBBN*rhoBBN*He3tanp_b*Yn1p0*Yn0p1*Yn2p2 + rhoBBN*He3tad_b*Yn1p1*Yn2p2 - rhoBBN*taLi7g_f*Yn2p1*Yn2p2 - rhoBBN*Li6nta_b*Yn2p1*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li6taan_b*Yn1p0*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*Li7taann_b*Yn1p0*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*rhoBBN*Be7taanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Be7taad_b*Yn1p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*Li8He3aat_b*Yn2p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*B8taaHe3_b*Yn1p2*Yn2p2*Yn2p2 + He3tLi6g_b*Yn3p3 + rhoBBN*Li6nta_f*Yn1p0*Yn3p3 - rhoBBN*Li6taan_f*Yn2p1*Yn3p3 - rhoBBN*Li6tLi8p_f*Yn2p1*Yn3p3 - rhoBBN*Li6tLi7d_f*Yn2p1*Yn3p3 + rhoBBN*Be7tLi6a_b*Yn2p2*Yn3p3 + taLi7g_b*Yn4p3 + rhoBBN*Li6tLi7d_b*Yn1p1*Yn4p3 - rhoBBN*Li7taann_f*Yn2p1*Yn4p3 - rhoBBN*Li8dLi7t_b*Yn2p1*Yn4p3 + rhoBBN*Be7tLi7He3_b*Yn1p2*Yn4p3 + rhoBBN*Li6tLi8p_b*Yn0p1*Yn5p3 + rhoBBN*Li8dLi7t_f*Yn1p1*Yn5p3 + rhoBBN*Li8He3aat_f*Yn1p2*Yn5p3 - rhoBBN*Be7tLi6a_f*Yn2p1*Yn3p4 - rhoBBN*Be7taad_f*Yn2p1*Yn3p4 - rhoBBN*Be7tLi7He3_f*Yn2p1*Yn3p4 - rhoBBN*Be7taanp_f*Yn2p1*Yn3p4 + rhoBBN*B8tBe7a_b*Yn2p2*Yn3p4 - rhoBBN*B8tBe7a_f*Yn2p1*Yn3p5 - rhoBBN*B8taaHe3_f*Yn2p1*Yn3p5
    dYHe3 = rhoBBN*dpHe3g_f*Yn0p1*Yn1p1 + 0.5*rhoBBN*ddHe3n_f*Yn1p1*Yn1p1 + rhoBBN*He3ntp_b*Yn0p1*Yn2p1 - dpHe3g_b*Yn1p2 - rhoBBN*He3ntp_f*Yn1p0*Yn1p2 - rhoBBN*He3nag_f*Yn1p0*Yn1p2 - rhoBBN*ddHe3n_b*Yn1p0*Yn1p2 - rhoBBN*He3dap_f*Yn1p1*Yn1p2 - rhoBBN*He3tad_f*Yn2p1*Yn1p2 - rhoBBN*He3tanp_f*Yn2p1*Yn1p2 - rhoBBN*He3tLi6g_f*Yn2p1*Yn1p2 - rhoBBN*He3He3app_f*Yn1p2*Yn1p2 + He3nag_b*Yn2p2 + rhoBBN*He3dap_b*Yn0p1*Yn2p2 + rhoBBN*rhoBBN*He3tanp_b*Yn1p0*Yn0p1*Yn2p2 + rhoBBN*rhoBBN*He3He3app_b*Yn0p1*Yn0p1*Yn2p2 + rhoBBN*He3tad_b*Yn1p1*Yn2p2 - rhoBBN*He3aBe7g_f*Yn1p2*Yn2p2 - rhoBBN*Li6pHe3a_b*Yn1p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li6He3aap_b*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*rhoBBN*Li7He3aanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*Be7He3ppaa_b*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*Be7He3aapp_b*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li7He3aad_b*Yn1p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li8He3aat_b*Yn2p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*B8taaHe3_b*Yn1p2*Yn2p2*Yn2p2 + He3tLi6g_b*Yn3p3 + rhoBBN*Li6pHe3a_f*Yn0p1*Yn3p3 - rhoBBN*Li6He3aap_f*Yn1p2*Yn3p3 - rhoBBN*Li6He3Be7d_f*Yn1p2*Yn3p3 - rhoBBN*B8nLi6He3_b*Yn1p2*Yn3p3 + rhoBBN*Li7He3Li6a_b*Yn2p2*Yn3p3 - rhoBBN*Li7He3Li6a_f*Yn1p2*Yn4p3 - rhoBBN*Li7He3aad_f*Yn1p2*Yn4p3 - rhoBBN*Li7He3aanp_f*Yn1p2*Yn4p3 - rhoBBN*Be7tLi7He3_b*Yn1p2*Yn4p3 + rhoBBN*Li8He3Li7a_b*Yn2p2*Yn4p3 - rhoBBN*Li8He3Li7a_f*Yn1p2*Yn5p3 - rhoBBN*Li8He3aat_f*Yn1p2*Yn5p3 + He3aBe7g_b*Yn3p4 + rhoBBN*Li6He3Be7d_b*Yn1p1*Yn3p4 + rhoBBN*Be7tLi7He3_f*Yn2p1*Yn3p4 - rhoBBN*Be7He3ppaa_f*Yn1p2*Yn3p4 - rhoBBN*Be7He3aapp_f*Yn1p2*Yn3p4 - rhoBBN*B8dBe7He3_b*Yn1p2*Yn3p4 + rhoBBN*B8nLi6He3_f*Yn1p0*Yn3p5 + rhoBBN*B8dBe7He3_f*Yn1p1*Yn3p5 + rhoBBN*B8taaHe3_f*Yn2p1*Yn3p5
    dYa = 0.5*rhoBBN*ddag_f*Yn1p1*Yn1p1 + rhoBBN*tpag_f*Yn0p1*Yn2p1 + rhoBBN*tdan_f*Yn1p1*Yn2p1 + 0.5*rhoBBN*ttann_f*Yn2p1*Yn2p1 + rhoBBN*He3nag_f*Yn1p0*Yn1p2 + rhoBBN*He3dap_f*Yn1p1*Yn1p2 + rhoBBN*He3tad_f*Yn2p1*Yn1p2 + rhoBBN*He3tanp_f*Yn2p1*Yn1p2 + 0.5*rhoBBN*He3He3app_f*Yn1p2*Yn1p2 - tpag_b*Yn2p2 - ddag_b*Yn2p2 - He3nag_b*Yn2p2 - rhoBBN*tdan_b*Yn1p0*Yn2p2 - 0.5*rhoBBN*rhoBBN*annHe6g_f*Yn1p0*Yn1p0*Yn2p2 - 0.5*rhoBBN*rhoBBN*ttann_b*Yn1p0*Yn1p0*Yn2p2 - rhoBBN*He3dap_b*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*anpLi6g_f*Yn1p0*Yn0p1*Yn2p2 - rhoBBN*rhoBBN*He3tanp_b*Yn1p0*Yn0p1*Yn2p2 - 0.5*rhoBBN*rhoBBN*He3He3app_b*Yn0p1*Yn0p1*Yn2p2 - rhoBBN*daLi6g_f*Yn1p1*Yn2p2 - rhoBBN*He3tad_b*Yn1p1*Yn2p2 - rhoBBN*taLi7g_f*Yn2p1*Yn2p2 - rhoBBN*Li6nta_b*Yn2p1*Yn2p2 - rhoBBN*He3aBe7g_f*Yn1p2*Yn2p2 - rhoBBN*Li6pHe3a_b*Yn1p2*Yn2p2 - rhoBBN*Li7paa_b*Yn2p2*Yn2p2 - rhoBBN*Li7paag_b*Yn2p2*Yn2p2 - rhoBBN*Be7naa_b*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*Li6taan_b*Yn1p0*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*Li7daan_b*Yn1p0*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*Li8paan_b*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Li7taann_b*Yn1p0*Yn1p0*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*B8naap_b*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*Li6He3aap_b*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*rhoBBN*Li7He3aanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*rhoBBN*Be7taanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Be7He3ppaa_b*Yn0p1*Yn0p1*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*rhoBBN*Be7He3aapp_b*Yn0p1*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*Li7He3aad_b*Yn1p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*Be7taad_b*Yn1p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*Li8He3aat_b*Yn2p1*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*B8taaHe3_b*Yn1p2*Yn2p2*Yn2p2 + annHe6g_b*Yn4p2 + daLi6g_b*Yn3p3 + anpLi6g_b*Yn3p3 + rhoBBN*Li6nta_f*Yn1p0*Yn3p3 + rhoBBN*Li6pHe3a_f*Yn0p1*Yn3p3 + 2*rhoBBN*Li6taan_f*Yn2p1*Yn3p3 + 2*rhoBBN*Li6He3aap_f*Yn1p2*Yn3p3 - rhoBBN*Li7He3Li6a_b*Yn2p2*Yn3p3 - rhoBBN*Be7tLi6a_b*Yn2p2*Yn3p3 + taLi7g_b*Yn4p3 + 2*rhoBBN*Li7paa_f*Yn0p1*Yn4p3 + 2*rhoBBN*Li7paag_f*Yn0p1*Yn4p3 + 2*rhoBBN*Li7daan_f*Yn1p1*Yn4p3 + 2*rhoBBN*Li7taann_f*Yn2p1*Yn4p3 + rhoBBN*Li7He3Li6a_f*Yn1p2*Yn4p3 + 2*rhoBBN*Li7He3aad_f*Yn1p2*Yn4p3 + 2*rhoBBN*Li7He3aanp_f*Yn1p2*Yn4p3 - rhoBBN*Li8He3Li7a_b*Yn2p2*Yn4p3 + 2*rhoBBN*Li8paan_f*Yn0p1*Yn5p3 + rhoBBN*Li8He3Li7a_f*Yn1p2*Yn5p3 + 2*rhoBBN*Li8He3aat_f*Yn1p2*Yn5p3 + He3aBe7g_b*Yn3p4 + 2*rhoBBN*Be7naa_f*Yn1p0*Yn3p4 + 2*rhoBBN*Be7daap_f*Yn1p1*Yn3p4 + rhoBBN*Be7tLi6a_f*Yn2p1*Yn3p4 + 2*rhoBBN*Be7taad_f*Yn2p1*Yn3p4 + 2*rhoBBN*Be7taanp_f*Yn2p1*Yn3p4 + 2*rhoBBN*Be7He3ppaa_f*Yn1p2*Yn3p4 + 2*rhoBBN*Be7He3aapp_f*Yn1p2*Yn3p4 - rhoBBN*B8tBe7a_b*Yn2p2*Yn3p4 + 2*rhoBBN*B8naap_f*Yn1p0*Yn3p5 + rhoBBN*B8tBe7a_f*Yn2p1*Yn3p5 + 2*rhoBBN*B8taaHe3_f*Yn2p1*Yn3p5
    dYHe6 = 0.5*rhoBBN*rhoBBN*annHe6g_f*Yn1p0*Yn1p0*Yn2p2 - annHe6g_b*Yn4p2
    dYLi6 = rhoBBN*He3tLi6g_f*Yn2p1*Yn1p2 + rhoBBN*rhoBBN*anpLi6g_f*Yn1p0*Yn0p1*Yn2p2 + rhoBBN*daLi6g_f*Yn1p1*Yn2p2 + rhoBBN*Li6nta_b*Yn2p1*Yn2p2 + rhoBBN*Li6pHe3a_b*Yn1p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li6taan_b*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li6He3aap_b*Yn0p1*Yn2p2*Yn2p2 - daLi6g_b*Yn3p3 - He3tLi6g_b*Yn3p3 - anpLi6g_b*Yn3p3 - rhoBBN*Li6nta_f*Yn1p0*Yn3p3 - rhoBBN*Li6nLi7g_f*Yn1p0*Yn3p3 - rhoBBN*Li6pBe7g_f*Yn0p1*Yn3p3 - rhoBBN*Li6pHe3a_f*Yn0p1*Yn3p3 - rhoBBN*Li6dLi7p_f*Yn1p1*Yn3p3 - rhoBBN*Li6dBe7n_f*Yn1p1*Yn3p3 - rhoBBN*Li6taan_f*Yn2p1*Yn3p3 - rhoBBN*Li6tLi8p_f*Yn2p1*Yn3p3 - rhoBBN*Li6tLi7d_f*Yn2p1*Yn3p3 - rhoBBN*Li6He3aap_f*Yn1p2*Yn3p3 - rhoBBN*Li6He3Be7d_f*Yn1p2*Yn3p3 - rhoBBN*B8nLi6He3_b*Yn1p2*Yn3p3 - rhoBBN*Li7He3Li6a_b*Yn2p2*Yn3p3 - rhoBBN*Be7tLi6a_b*Yn2p2*Yn3p3 + Li6nLi7g_b*Yn4p3 + rhoBBN*Li6dLi7p_b*Yn0p1*Yn4p3 + rhoBBN*Li6tLi7d_b*Yn1p1*Yn4p3 + rhoBBN*Li7He3Li6a_f*Yn1p2*Yn4p3 + rhoBBN*Li6tLi8p_b*Yn0p1*Yn5p3 + Li6pBe7g_b*Yn3p4 + rhoBBN*Li6dBe7n_b*Yn1p0*Yn3p4 + rhoBBN*Li6He3Be7d_b*Yn1p1*Yn3p4 + rhoBBN*Be7tLi6a_f*Yn2p1*Yn3p4 + rhoBBN*B8nLi6He3_f*Yn1p0*Yn3p5
    dYLi7 = rhoBBN*taLi7g_f*Yn2p1*Yn2p2 + 0.5*rhoBBN*Li7paa_b*Yn2p2*Yn2p2 + 0.5*rhoBBN*Li7paag_b*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li7daan_b*Yn1p0*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*Li7taann_b*Yn1p0*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*rhoBBN*Li7He3aanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li7He3aad_b*Yn1p1*Yn2p2*Yn2p2 + rhoBBN*Li6nLi7g_f*Yn1p0*Yn3p3 + rhoBBN*Li6dLi7p_f*Yn1p1*Yn3p3 + rhoBBN*Li6tLi7d_f*Yn2p1*Yn3p3 + rhoBBN*Li7He3Li6a_b*Yn2p2*Yn3p3 - taLi7g_b*Yn4p3 - Li6nLi7g_b*Yn4p3 - rhoBBN*Li7nLi8g_f*Yn1p0*Yn4p3 - rhoBBN*Li7paa_f*Yn0p1*Yn4p3 - rhoBBN*Li7paag_f*Yn0p1*Yn4p3 - rhoBBN*Be7nLi7p_b*Yn0p1*Yn4p3 - rhoBBN*Li6dLi7p_b*Yn0p1*Yn4p3 - rhoBBN*Li7daan_f*Yn1p1*Yn4p3 - rhoBBN*Li7dLi8p_f*Yn1p1*Yn4p3 - rhoBBN*Li6tLi7d_b*Yn1p1*Yn4p3 - rhoBBN*Li7taann_f*Yn2p1*Yn4p3 - rhoBBN*Li8dLi7t_b*Yn2p1*Yn4p3 - rhoBBN*Li7He3Li6a_f*Yn1p2*Yn4p3 - rhoBBN*Li7He3aad_f*Yn1p2*Yn4p3 - rhoBBN*Li7He3aanp_f*Yn1p2*Yn4p3 - rhoBBN*Be7tLi7He3_b*Yn1p2*Yn4p3 - rhoBBN*Li8He3Li7a_b*Yn2p2*Yn4p3 + Li7nLi8g_b*Yn5p3 + rhoBBN*Li7dLi8p_b*Yn0p1*Yn5p3 + rhoBBN*Li8dLi7t_f*Yn1p1*Yn5p3 + rhoBBN*Li8He3Li7a_f*Yn1p2*Yn5p3 + rhoBBN*Be7nLi7p_f*Yn1p0*Yn3p4 + rhoBBN*Be7tLi7He3_f*Yn2p1*Yn3p4
    dYLi8 = 0.5*rhoBBN*rhoBBN*Li8paan_b*Yn1p0*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Li8He3aat_b*Yn2p1*Yn2p2*Yn2p2 + rhoBBN*Li6tLi8p_f*Yn2p1*Yn3p3 + rhoBBN*Li7nLi8g_f*Yn1p0*Yn4p3 + rhoBBN*Li7dLi8p_f*Yn1p1*Yn4p3 + rhoBBN*Li8dLi7t_b*Yn2p1*Yn4p3 + rhoBBN*Li8He3Li7a_b*Yn2p2*Yn4p3 - Li7nLi8g_b*Yn5p3 - rhoBBN*Li8paan_f*Yn0p1*Yn5p3 - rhoBBN*Li6tLi8p_b*Yn0p1*Yn5p3 - rhoBBN*Li7dLi8p_b*Yn0p1*Yn5p3 - rhoBBN*Li8dLi7t_f*Yn1p1*Yn5p3 - rhoBBN*Li8He3Li7a_f*Yn1p2*Yn5p3 - rhoBBN*Li8He3aat_f*Yn1p2*Yn5p3
    dYBe7 = rhoBBN*He3aBe7g_f*Yn1p2*Yn2p2 + 0.5*rhoBBN*Be7naa_b*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*rhoBBN*Be7taanp_b*Yn1p0*Yn0p1*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*Be7He3ppaa_b*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + 0.25*rhoBBN*rhoBBN*rhoBBN*Be7He3aapp_b*Yn0p1*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*Be7taad_b*Yn1p1*Yn2p2*Yn2p2 + rhoBBN*Li6pBe7g_f*Yn0p1*Yn3p3 + rhoBBN*Li6dBe7n_f*Yn1p1*Yn3p3 + rhoBBN*Li6He3Be7d_f*Yn1p2*Yn3p3 + rhoBBN*Be7tLi6a_b*Yn2p2*Yn3p3 + rhoBBN*Be7nLi7p_b*Yn0p1*Yn4p3 + rhoBBN*Be7tLi7He3_b*Yn1p2*Yn4p3 - He3aBe7g_b*Yn3p4 - Li6pBe7g_b*Yn3p4 - rhoBBN*Be7nLi7p_f*Yn1p0*Yn3p4 - rhoBBN*Be7naa_f*Yn1p0*Yn3p4 - rhoBBN*Li6dBe7n_b*Yn1p0*Yn3p4 - rhoBBN*Be7pB8g_f*Yn0p1*Yn3p4 - rhoBBN*Be7daap_f*Yn1p1*Yn3p4 - rhoBBN*B8nBe7d_b*Yn1p1*Yn3p4 - rhoBBN*Li6He3Be7d_b*Yn1p1*Yn3p4 - rhoBBN*Be7tLi6a_f*Yn2p1*Yn3p4 - rhoBBN*Be7taad_f*Yn2p1*Yn3p4 - rhoBBN*Be7tLi7He3_f*Yn2p1*Yn3p4 - rhoBBN*Be7taanp_f*Yn2p1*Yn3p4 - rhoBBN*Be7He3ppaa_f*Yn1p2*Yn3p4 - rhoBBN*Be7He3aapp_f*Yn1p2*Yn3p4 - rhoBBN*B8dBe7He3_b*Yn1p2*Yn3p4 - rhoBBN*B8tBe7a_b*Yn2p2*Yn3p4 + Be7pB8g_b*Yn3p5 + rhoBBN*B8nBe7d_f*Yn1p0*Yn3p5 + rhoBBN*B8dBe7He3_f*Yn1p1*Yn3p5 + rhoBBN*B8tBe7a_f*Yn2p1*Yn3p5
    dYB8 = 0.5*rhoBBN*rhoBBN*B8naap_b*Yn0p1*Yn2p2*Yn2p2 + 0.5*rhoBBN*rhoBBN*B8taaHe3_b*Yn1p2*Yn2p2*Yn2p2 + rhoBBN*B8nLi6He3_b*Yn1p2*Yn3p3 + rhoBBN*Be7pB8g_f*Yn0p1*Yn3p4 + rhoBBN*B8nBe7d_b*Yn1p1*Yn3p4 + rhoBBN*B8dBe7He3_b*Yn1p2*Yn3p4 + rhoBBN*B8tBe7a_b*Yn2p2*Yn3p4 - Be7pB8g_b*Yn3p5 - rhoBBN*B8naap_f*Yn1p0*Yn3p5 - rhoBBN*B8nLi6He3_f*Yn1p0*Yn3p5 - rhoBBN*B8nBe7d_f*Yn1p0*Yn3p5 - rhoBBN*B8dBe7He3_f*Yn1p1*Yn3p5 - rhoBBN*B8tBe7a_f*Yn2p1*Yn3p5 - rhoBBN*B8taaHe3_f*Yn2p1*Yn3p5
    return (dYn, dYp, dYd, dYt, dYHe3, dYa, dYLi7, dYBe7, dYHe6, dYLi8, dYLi6, dYB8)


_jac_impl   = None
_jacMT_impl = None
_jacLT_impl = None


def _setup_nuclear_rhs_impls(numba_flag):
    global _rhs_impl, _rhsMT_impl, _rhsLT_impl
    global _jac_impl, _jacMT_impl, _jacLT_impl
    if _rhs_impl is not None:
        return
    if numba_flag:
        try:
            from numba import njit
            _rhs_impl   = njit(_rhs_arith)
            _rhsMT_impl = njit(_rhsMT_arith)
            _rhsLT_impl = njit(_rhsLT_arith)
            _jac_impl   = njit(_jac_arith)
            _jacMT_impl = njit(_jacMT_arith)
            _jacLT_impl = njit(_jacLT_arith)
            return
        except ImportError:
            pass
    _rhs_impl   = _rhs_arith
    _rhsMT_impl = _rhsMT_arith
    _rhsLT_impl = _rhsLT_arith
    _jac_impl   = _jac_arith
    _jacMT_impl = _jacMT_arith
    _jacLT_impl = _jacLT_arith

def _jac_arith(Y, rhoBBN, r):
    Yn, Yp, Yd, Yt, YHe3, Ya, YLi7, YBe7 = Y

    nTOp_f = r[0]; nTOp_b = r[1]
    npdg_f = r[2]; npdg_b = r[3]
    dpHe3g_f = r[4]; dpHe3g_b = r[5]
    ddHe3n_f = r[6]; ddHe3n_b = r[7]
    ddtp_f = r[8]; ddtp_b = r[9]
    tpag_f = r[10]; tpag_b = r[11]
    tdan_f = r[12]; tdan_b = r[13]
    taLi7g_f = r[14]; taLi7g_b = r[15]
    He3ntp_f = r[16]; He3ntp_b = r[17]
    He3dap_f = r[18]; He3dap_b = r[19]
    He3aBe7g_f = r[20]; He3aBe7g_b = r[21]
    Be7nLi7p_f = r[22]; Be7nLi7p_b = r[23]
    Li7paa_f = r[24]; Li7paa_b = r[25]

    J = np.empty((8, 8))
    J[0, 0] = -nTOp_f + rhoBBN*(-npdg_f*Yp - (He3ntp_f+ddHe3n_b)*YHe3 - tdan_b*Ya - Be7nLi7p_f*YBe7)
    J[0, 1] = nTOp_b + rhoBBN*(He3ntp_b*Yt - npdg_f*Yn + Be7nLi7p_b*YLi7)
    J[0, 2] = rhoBBN*(ddHe3n_f*Yd + tdan_f*Yt) + npdg_b
    J[0, 3] = rhoBBN*(He3ntp_b*Yp + tdan_f*Yd)
    J[0, 4] = -rhoBBN*(He3ntp_f+ddHe3n_b)*Yn
    J[0, 5] = -rhoBBN*tdan_b*Yn
    J[0, 6] = rhoBBN*Be7nLi7p_b*Yp
    J[0, 7] = -rhoBBN*Be7nLi7p_f*Yn
    J[1, 0] = nTOp_f + rhoBBN*(-npdg_f*Yp + He3ntp_f*YHe3 + Be7nLi7p_f*YBe7)
    J[1, 1] = -nTOp_b + rhoBBN*(-npdg_f*Yn - dpHe3g_f*Yd - (tpag_f+ddtp_b+He3ntp_b)*Yt - He3dap_b*Ya - (Li7paa_f+Be7nLi7p_b)*YLi7)
    J[1, 2] = rhoBBN*(ddtp_f*Yd - dpHe3g_f*Yp + He3dap_f*YHe3) + npdg_b
    J[1, 3] = -rhoBBN*(tpag_f+ddtp_b+He3ntp_b)*Yp
    J[1, 4] = rhoBBN*(He3ntp_f*Yn + He3dap_f*Yd) + dpHe3g_b
    J[1, 5] = rhoBBN*(-He3dap_b*Yp + Li7paa_b*Ya) + tpag_b
    J[1, 6] = rhoBBN*Be7nLi7p_f*Yn
    J[1, 7] = rhoBBN*Be7nLi7p_f*Yn
    J[2, 0] = rhoBBN*(npdg_f*Yp + 2.*ddHe3n_b*YHe3 + tdan_b*Ya)
    J[2, 1] = rhoBBN*(npdg_f*Yn - dpHe3g_f*Yd + 2.*ddtp_b*Yt + He3dap_b*Ya)
    J[2, 2] = rhoBBN*(-dpHe3g_f*Yp - 2.*(ddHe3n_f+ddtp_f)*Yd - tdan_f*Yt - He3dap_f*YHe3) - npdg_b
    J[2, 3] = rhoBBN*(2.*ddtp_b*Yp - tdan_f*Yd)
    J[2, 4] = rhoBBN*(2.*ddHe3n_b*Yn - He3dap_f*Yd) + dpHe3g_b
    J[2, 5] = rhoBBN*(tdan_b*Yn + He3dap_b*Yp)
    J[2, 6] = 0.
    J[2, 7] = 0.
    J[3, 0] = rhoBBN*(He3ntp_f*YHe3 + tdan_b*Ya)
    J[3, 1] = -rhoBBN*(tpag_f+ddtp_b+He3ntp_b)*Yt
    J[3, 2] = rhoBBN*(ddtp_f*Yd - tdan_f*Yt)
    J[3, 3] = -rhoBBN*((tpag_f+ddtp_b+He3ntp_b)*Yp + tdan_f*Yd + taLi7g_f*Ya)
    J[3, 4] = rhoBBN*He3ntp_f*Yn
    J[3, 5] = rhoBBN*(tdan_b*Yn - taLi7g_f*Yt) + tpag_b
    J[3, 6] = taLi7g_b
    J[3, 7] = 0.
    J[4, 0] = -rhoBBN*(He3ntp_f+ddHe3n_b)*YHe3
    J[4, 1] = rhoBBN*(dpHe3g_f*Yd + He3ntp_b*Yt + He3dap_b*Ya)
    J[4, 2] = rhoBBN*(dpHe3g_f*Yp + ddHe3n_f*Yd - He3dap_f*YHe3)
    J[4, 3] = rhoBBN*He3ntp_b*Yp
    J[4, 4] = rhoBBN*(-He3dap_f*Yd - (He3ntp_f+ddHe3n_b)*Yn - He3aBe7g_f*Ya) - dpHe3g_b
    J[4, 5] = rhoBBN*(He3dap_b*Yp - He3aBe7g_f*YHe3)
    J[4, 6] = 0.
    J[4, 7] = He3aBe7g_b
    J[5, 0] = -rhoBBN*tdan_b*Ya
    J[5, 1] = rhoBBN*(-He3dap_b*Ya + 2.*Li7paa_f*YLi7 + tpag_f*Yt)
    J[5, 2] = rhoBBN*(He3dap_f*YHe3 + tdan_f*Yt)
    J[5, 3] = rhoBBN*(-taLi7g_f*Ya + tdan_f*Yd + tpag_f*Yp)
    J[5, 4] = rhoBBN*(-He3aBe7g_f*Ya + He3dap_f*Yd)
    J[5, 5] = -rhoBBN*(He3aBe7g_f*YHe3 + He3dap_b*Yp + 2.*Li7paa_b*Ya + taLi7g_f*Yt + tdan_b*Yn) - tpag_b
    J[5, 6] = 2.*rhoBBN*Li7paa_f*Yp + taLi7g_b
    J[5, 7] = He3aBe7g_b
    J[6, 0] = rhoBBN*Be7nLi7p_f*YBe7
    J[6, 1] = -rhoBBN*(Be7nLi7p_b+Li7paa_f)*YLi7
    J[6, 2] = 0.
    J[6, 3] = rhoBBN*taLi7g_f*Ya
    J[6, 4] = 0.
    J[6, 5] = rhoBBN*(Li7paa_b*Ya + taLi7g_f*Yt)
    J[6, 6] = -rhoBBN*(Be7nLi7p_b+Li7paa_f)*Yp - taLi7g_b
    J[6, 7] = rhoBBN*Be7nLi7p_f*Yn
    J[7, 0] = -rhoBBN*Be7nLi7p_f*YBe7
    J[7, 1] = rhoBBN*Be7nLi7p_b*YLi7
    J[7, 2] = 0.
    J[7, 3] = 0.
    J[7, 4] = rhoBBN*He3aBe7g_f*Ya
    J[7, 5] = rhoBBN*He3aBe7g_f*YHe3
    J[7, 6] = rhoBBN*Be7nLi7p_b*Yp
    J[7, 7] = -rhoBBN*Be7nLi7p_f*Yn - He3aBe7g_b
    return J


def _jacMT_arith(Y, rhoBBN, r):
    Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y

    nTOp_f = r[0]; nTOp_b = r[1]
    Be7daap_f = r[2]; Be7daap_b = r[3]
    Be7nLi7p_f = r[4]; Be7nLi7p_b = r[5]
    Be7naa_f = r[6]; Be7naa_b = r[7]
    He3aBe7g_f = r[8]; He3aBe7g_b = r[9]
    He3dap_f = r[10]; He3dap_b = r[11]
    He3ntp_f = r[12]; He3ntp_b = r[13]
    Li6pBe7g_f = r[14]; Li6pBe7g_b = r[15]
    Li7paa_f = r[16]; Li7paa_b = r[17]
    Li7paag_f = r[18]; Li7paag_b = r[19]
    daLi6g_f = r[20]; daLi6g_b = r[21]
    ddHe3n_f = r[22]; ddHe3n_b = r[23]
    ddtp_f = r[24]; ddtp_b = r[25]
    dpHe3g_f = r[26]; dpHe3g_b = r[27]
    npdg_f = r[28]; npdg_b = r[29]
    taLi7g_f = r[30]; taLi7g_b = r[31]
    tdan_f = r[32]; tdan_b = r[33]
    tpag_f = r[34]; tpag_b = r[35]

    # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4}
    # Yn
    dYn_primeOdYn = rhoBBN*(-npdg_f*Yn0p1 - (He3ntp_f + ddHe3n_b)*Yn1p2 - tdan_b*Yn2p2 - Be7nLi7p_f*Yn3p4) + rhoBBN*(-Be7naa_f*Yn3p4)
    dYn_primeOdYp = nTOp_b + rhoBBN*(- npdg_f*Yn1p0 + He3ntp_b*Yn2p1 + Be7nLi7p_b*Yn4p3)
    dYn_primeOdYd = rhoBBN*(ddHe3n_f*Yn1p1 + tdan_f*Yn2p1) + npdg_b
    dYn_primeOdYt = rhoBBN*(He3ntp_b*Yn0p1 + tdan_f*Yn1p1)
    dYn_primeOdYHe3 = -rhoBBN*(He3ntp_f + ddHe3n_b)*Yn1p0
    dYn_primeOdYa = -rhoBBN*tdan_b*Yn1p0 + rhoBBN*(0.5*Be7naa_b*Yn2p2*2)
    dYn_primeOdYLi7 = rhoBBN*Be7nLi7p_b*Yn0p1
    dYn_primeOdYBe7 = -rhoBBN*Be7nLi7p_f*Yn1p0 + rhoBBN*(-Be7naa_f*Yn1p0)
    dYn_primeOdYHe6 = 0.
    dYn_primeOdYLi8 = 0.
    dYn_primeOdYLi6 = 0.
    dYn_primeOdYB8 = 0.

    # Yp
    dYp_primeOdYn = nTOp_f + rhoBBN*(- npdg_f*Yn0p1 + He3ntp_f*Yn1p2 + Be7nLi7p_f*Yn3p4)
    dYp_primeOdYp = -nTOp_b*Yn0p1 + rhoBBN*(- npdg_f*Yn1p0 - dpHe3g_f*Yn1p1 - (tpag_f + ddtp_b + He3ntp_b)*Yn2p1 - He3dap_b*Yn2p2 - (Li7paa_f + Be7nLi7p_b)*Yn4p3) + rhoBBN*(- 0.5*rhoBBN*Be7daap_b*Yn2p2*Yn2p2) + rhoBBN*(-Li6pBe7g_f*Yn3p3) + rhoBBN*(-Li7paag_f*Yn4p3)
    dYp_primeOdYd = rhoBBN*(ddtp_f*Yn1p1 - dpHe3g_f*Yn0p1 + He3dap_f*Yn1p2) + npdg_b + rhoBBN*(Be7daap_f*Yn3p4)
    dYp_primeOdYt = -rhoBBN*(tpag_f + ddtp_b + He3ntp_b)*Yn0p1
    dYp_primeOdYHe3 = rhoBBN*(He3ntp_f*Yn1p0 + He3dap_f*Yn1p1) + dpHe3g_b
    dYp_primeOdYa = rhoBBN*(-He3dap_b*Yn0p1 + Li7paa_b*Yn2p2) + tpag_b + rhoBBN*(- 0.5*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*2) + 0.5*rhoBBN*Li7paag_b*Yn2p2*2
    dYp_primeOdYLi7 = rhoBBN*(-(Li7paa_f + Be7nLi7p_b)*Yn0p1) + rhoBBN*(-Li7paag_f*Yn0p1)
    dYp_primeOdYBe7 = rhoBBN*Be7nLi7p_f*Yn1p0 + rhoBBN*(Be7daap_f*Yn1p1) + (Li6pBe7g_b)
    dYp_primeOdYHe6 = 0.
    dYp_primeOdYLi8 = 0.
    dYp_primeOdYLi6 = rhoBBN*(-Li6pBe7g_f*Yn0p1)
    dYp_primeOdYB8 = 0.

    # Yd
    dYd_primeOdYn = rhoBBN*(npdg_f*Yn0p1 + 2.*ddHe3n_b*Yn1p2 + tdan_b*Yn2p2)
    dYd_primeOdYp = rhoBBN*(npdg_f*Yn1p0 - dpHe3g_f*Yn1p1 + 2.*ddtp_b*Yn2p1 + He3dap_b*Yn2p2) + rhoBBN*(0.5*rhoBBN*Be7daap_b*Yn2p2*Yn2p2)
    dYd_primeOdYd = rhoBBN*(- dpHe3g_f*Yn0p1 - (ddtp_f)*Yn1p1*2 - tdan_f*Yn2p1 - He3dap_f*Yn1p2) - npdg_b + rhoBBN*(-Be7daap_f*Yn3p4) + rhoBBN*(-daLi6g_f*Yn2p2)
    dYd_primeOdYt = rhoBBN*(2.*ddtp_b*Yn0p1 - tdan_f*Yn1p1)
    dYd_primeOdYHe3 = rhoBBN*(2.*ddHe3n_b*Yn1p0 - He3dap_f*Yn1p1) + dpHe3g_b
    dYd_primeOdYa = rhoBBN*(tdan_b*Yn1p0 + He3dap_b*Yn0p1) + rhoBBN*(0.5*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*2) + rhoBBN*(-daLi6g_f*Yn1p1)
    dYd_primeOdYLi7 = 0.
    dYd_primeOdYBe7 = rhoBBN*(-Be7daap_f*Yn1p1)
    dYd_primeOdYHe6 = 0.
    dYd_primeOdYLi8 = 0.
    dYd_primeOdYLi6 = daLi6g_b
    dYd_primeOdYB8 = 0.

    # Yt
    dYt_primeOdYn = rhoBBN*(He3ntp_f*Yn1p2 + tdan_b*Yn2p2)
    dYt_primeOdYp = -rhoBBN*(tpag_f+ddtp_b+He3ntp_b)*Yn2p1
    dYt_primeOdYd = rhoBBN*(ddtp_f*Yn1p1 - tdan_f*Yn2p1)
    dYt_primeOdYt = -rhoBBN*((tpag_f + ddtp_b + He3ntp_b)*Yn0p1 + tdan_f*Yn1p1 + taLi7g_f*Yn2p2)
    dYt_primeOdYHe3 = rhoBBN*He3ntp_f*Yn1p0
    dYt_primeOdYa = rhoBBN*(tdan_b*Yn1p0 - taLi7g_f*Yn2p1) + tpag_b
    dYt_primeOdYLi7 = taLi7g_b
    dYt_primeOdYBe7 = 0.
    dYt_primeOdYHe6 = 0.
    dYt_primeOdYLi8 = 0.
    dYt_primeOdYLi6 = 0.
    dYt_primeOdYB8 = 0.

    # YHe3
    dYHe3_primeOdYn = -rhoBBN*(He3ntp_f+ddHe3n_b)*Yn1p2
    dYHe3_primeOdYp = rhoBBN*(dpHe3g_f*Yn1p1 + He3ntp_b*Yn2p1 + He3dap_b*Yn2p2)
    dYHe3_primeOdYd = rhoBBN*(dpHe3g_f*Yn0p1 + ddHe3n_f*Yn1p1 - He3dap_f*Yn1p2)
    dYHe3_primeOdYt = rhoBBN*(He3ntp_b*Yn0p1)
    dYHe3_primeOdYHe3 = rhoBBN*(- He3dap_f*Yn1p1 - (He3ntp_f+ddHe3n_b)*Yn1p0 - He3aBe7g_f*Yn2p2) - dpHe3g_b
    dYHe3_primeOdYa = rhoBBN*(He3dap_b*Yn0p1 - He3aBe7g_f*Yn1p2)
    dYHe3_primeOdYLi7 = 0.
    dYHe3_primeOdYBe7 = He3aBe7g_b
    dYHe3_primeOdYHe6 = 0.
    dYHe3_primeOdYLi8 = 0
    dYHe3_primeOdYLi6 = 0
    dYHe3_primeOdYB8 = 0

    # Ya
    dYa_primeOdYn = -rhoBBN*tdan_b*Yn2p2 + rhoBBN*(2*Be7naa_f*Yn3p4)
    dYa_primeOdYp = rhoBBN*(- He3dap_b*Yn2p2 + 2.*Li7paa_f*Yn4p3 + tpag_f*Yn2p1) + rhoBBN*(- rhoBBN*Be7daap_b*Yn2p2*Yn2p2) + rhoBBN*(2*Li7paag_f*Yn4p3)
    dYa_primeOdYd = rhoBBN*(He3dap_f*Yn1p2 + tdan_f*Yn2p1) + rhoBBN*(2*Be7daap_f*Yn3p4) + rhoBBN*(-daLi6g_f*Yn2p2)
    dYa_primeOdYt = rhoBBN*(-taLi7g_f*Yn2p2 + tdan_f*Yn1p1 + tpag_f*Yn0p1)
    dYa_primeOdYHe3 = rhoBBN*(- He3aBe7g_f*Yn2p2 + He3dap_f*Yn1p1)
    dYa_primeOdYa = -rhoBBN*(He3aBe7g_f*Yn1p2 + He3dap_b*Yn0p1 + 2.*Li7paa_b*Yn2p2 + taLi7g_f*Yn2p1 + tdan_b*Yn1p0) - tpag_b + rhoBBN*(-Be7naa_b*Yn2p2*2) + rhoBBN*(-rhoBBN*Be7daap_b*Yn2p2*2*Yn0p1) + rhoBBN*(-daLi6g_f*Yn1p1) + (-rhoBBN*Li7paag_b*Yn2p2*2)
    dYa_primeOdYLi7 = 2.*rhoBBN*Li7paa_f*Yn0p1+taLi7g_b + rhoBBN*(2*Li7paag_f*Yn0p1)
    dYa_primeOdYBe7 = He3aBe7g_b + rhoBBN*(2*Be7naa_f*Yn1p0) + rhoBBN*(2*Be7daap_f*Yn1p1)
    dYa_primeOdYHe6 = 0.
    dYa_primeOdYLi8 = 0.
    dYa_primeOdYLi6 = daLi6g_b
    dYa_primeOdYB8 = 0.

    # YLi7
    dYLi7_primeOdYn = rhoBBN*Be7nLi7p_f*Yn3p4
    dYLi7_primeOdYp = -rhoBBN*(Be7nLi7p_b + Li7paa_f)*Yn4p3 + rhoBBN*(-Li7paag_f*Yn4p3)
    dYLi7_primeOdYd = 0.
    dYLi7_primeOdYt = rhoBBN*taLi7g_f*Yn2p2
    dYLi7_primeOdYHe3 = 0.
    dYLi7_primeOdYa = rhoBBN*(Li7paa_b*Yn2p2 + taLi7g_f*Yn2p1) + 0.5*rhoBBN*Li7paag_b*Yn2p2*2
    dYLi7_primeOdYLi7 = -rhoBBN*(Be7nLi7p_b + Li7paa_f)*Yn0p1 - taLi7g_b + rhoBBN*(-Li7paag_f*Yn0p1)
    dYLi7_primeOdYBe7 = rhoBBN*Be7nLi7p_f*Yn1p0
    dYLi7_primeOdYHe6 = 0.
    dYLi7_primeOdYLi8 = 0.
    dYLi7_primeOdYLi6 = 0.
    dYLi7_primeOdYB8 = 0.

    # YBe7
    dYBe7_primeOdYn = -rhoBBN*Be7nLi7p_f*Yn3p4 + rhoBBN*(-Be7naa_f*Yn3p4)
    dYBe7_primeOdYp = rhoBBN*Be7nLi7p_b*Yn4p3 + rhoBBN*(0.5*rhoBBN*Be7daap_b*Yn2p2*Yn2p2) + rhoBBN*(Li6pBe7g_f*Yn3p3)
    dYBe7_primeOdYd = rhoBBN*(-Be7daap_f*Yn3p4)
    dYBe7_primeOdYt = 0.
    dYBe7_primeOdYHe3 = rhoBBN*He3aBe7g_f*Yn2p2
    dYBe7_primeOdYa = rhoBBN*He3aBe7g_f*Yn1p2 + rhoBBN*(0.5*Be7naa_b*Yn2p2*2 + 0.5*rhoBBN*Be7daap_b*Yn0p1*Yn2p2*2)
    dYBe7_primeOdYLi7 = rhoBBN* Be7nLi7p_b*Yn0p1
    dYBe7_primeOdYBe7 = -rhoBBN*Be7nLi7p_f*Yn1p0 - He3aBe7g_b + rhoBBN*(-Be7naa_f*Yn1p0) + rhoBBN*(-Be7daap_f*Yn1p1) + (-Li6pBe7g_b)
    dYBe7_primeOdYHe6 = 0.
    dYBe7_primeOdYLi8 = 0.
    dYBe7_primeOdYLi6 = rhoBBN*(Li6pBe7g_f*Yn0p1)
    dYBe7_primeOdYB8 = 0.

    # YHe6
    dYHe6_primeOdYn = 0.
    dYHe6_primeOdYp = 0.
    dYHe6_primeOdYd = 0.
    dYHe6_primeOdYt = 0.
    dYHe6_primeOdYHe3 = 0.
    dYHe6_primeOdYa = 0.
    dYHe6_primeOdYLi7 = 0.
    dYHe6_primeOdYBe7 = 0.
    dYHe6_primeOdYHe6 = 0.
    dYHe6_primeOdYLi8 = 0.
    dYHe6_primeOdYLi6 = 0.
    dYHe6_primeOdYB8 = 0.

    # YLi8
    dYLi8_primeOdYn = 0.
    dYLi8_primeOdYp = 0.
    dYLi8_primeOdYd = 0.
    dYLi8_primeOdYt = 0.
    dYLi8_primeOdYHe3 = 0.
    dYLi8_primeOdYa = 0.
    dYLi8_primeOdYLi7 = 0.
    dYLi8_primeOdYBe7 = 0.
    dYLi8_primeOdYHe6 = 0.
    dYLi8_primeOdYLi8 = 0.
    dYLi8_primeOdYLi6 = 0.
    dYLi8_primeOdYB8 = 0.

    # YLi6
    dYLi6_primeOdYn = 0.
    dYLi6_primeOdYp = rhoBBN*(-Li6pBe7g_f*Yn3p3)
    dYLi6_primeOdYd = rhoBBN*(daLi6g_f*Yn2p2)
    dYLi6_primeOdYt = 0.
    dYLi6_primeOdYHe3 = 0.
    dYLi6_primeOdYa = rhoBBN*(daLi6g_f*Yn1p1)
    dYLi6_primeOdYLi7 = 0.
    dYLi6_primeOdYBe7 = Li6pBe7g_b
    dYLi6_primeOdYHe6 = 0.
    dYLi6_primeOdYLi8 = 0.
    dYLi6_primeOdYLi6 = (-daLi6g_b) + rhoBBN*(-Li6pBe7g_f*Yn0p1)
    dYLi6_primeOdYB8 = 0.

    # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4, YHe6 -> Yn4p2, Li8 -> Yn5p3, Li6 -> Yn3p3, B8 -> Yn3p5}
    # YB8
    dYB8_primeOdYn = 0.
    dYB8_primeOdYp = 0.
    dYB8_primeOdYd = 0.
    dYB8_primeOdYt = 0.
    dYB8_primeOdYHe3 = 0.
    dYB8_primeOdYa = 0.
    dYB8_primeOdYLi7 = 0.
    dYB8_primeOdYBe7 = 0.
    dYB8_primeOdYHe6 = 0.
    dYB8_primeOdYLi8 = 0.
    dYB8_primeOdYLi6 = 0.
    dYB8_primeOdYB8 = 0.



    J = np.empty((12, 12))
    J[0, 0] = dYn_primeOdYn
    J[0, 1] = dYn_primeOdYp
    J[0, 2] = dYn_primeOdYd
    J[0, 3] = dYn_primeOdYt
    J[0, 4] = dYn_primeOdYHe3
    J[0, 5] = dYn_primeOdYa
    J[0, 6] = dYn_primeOdYLi7
    J[0, 7] = dYn_primeOdYBe7
    J[0, 8] = dYn_primeOdYHe6
    J[0, 9] = dYn_primeOdYLi8
    J[0, 10] = dYn_primeOdYLi6
    J[0, 11] = dYn_primeOdYB8
    J[1, 0] = dYp_primeOdYn
    J[1, 1] = dYp_primeOdYp
    J[1, 2] = dYp_primeOdYd
    J[1, 3] = dYp_primeOdYt
    J[1, 4] = dYp_primeOdYHe3
    J[1, 5] = dYp_primeOdYa
    J[1, 6] = dYp_primeOdYLi7
    J[1, 7] = dYp_primeOdYBe7
    J[1, 8] = dYp_primeOdYHe6
    J[1, 9] = dYp_primeOdYLi8
    J[1, 10] = dYp_primeOdYLi6
    J[1, 11] = dYp_primeOdYB8
    J[2, 0] = dYd_primeOdYn
    J[2, 1] = dYd_primeOdYp
    J[2, 2] = dYd_primeOdYd
    J[2, 3] = dYd_primeOdYt
    J[2, 4] = dYd_primeOdYHe3
    J[2, 5] = dYd_primeOdYa
    J[2, 6] = dYd_primeOdYLi7
    J[2, 7] = dYd_primeOdYBe7
    J[2, 8] = dYd_primeOdYHe6
    J[2, 9] = dYd_primeOdYLi8
    J[2, 10] = dYd_primeOdYLi6
    J[2, 11] = dYd_primeOdYB8
    J[3, 0] = dYt_primeOdYn
    J[3, 1] = dYt_primeOdYp
    J[3, 2] = dYt_primeOdYd
    J[3, 3] = dYt_primeOdYt
    J[3, 4] = dYt_primeOdYHe3
    J[3, 5] = dYt_primeOdYa
    J[3, 6] = dYt_primeOdYLi7
    J[3, 7] = dYt_primeOdYBe7
    J[3, 8] = dYt_primeOdYHe6
    J[3, 9] = dYt_primeOdYLi8
    J[3, 10] = dYt_primeOdYLi6
    J[3, 11] = dYt_primeOdYB8
    J[4, 0] = dYHe3_primeOdYn
    J[4, 1] = dYHe3_primeOdYp
    J[4, 2] = dYHe3_primeOdYd
    J[4, 3] = dYHe3_primeOdYt
    J[4, 4] = dYHe3_primeOdYHe3
    J[4, 5] = dYHe3_primeOdYa
    J[4, 6] = dYHe3_primeOdYLi7
    J[4, 7] = dYHe3_primeOdYBe7
    J[4, 8] = dYHe3_primeOdYHe6
    J[4, 9] = dYHe3_primeOdYLi8
    J[4, 10] = dYHe3_primeOdYLi6
    J[4, 11] = dYHe3_primeOdYB8
    J[5, 0] = dYa_primeOdYn
    J[5, 1] = dYa_primeOdYp
    J[5, 2] = dYa_primeOdYd
    J[5, 3] = dYa_primeOdYt
    J[5, 4] = dYa_primeOdYHe3
    J[5, 5] = dYa_primeOdYa
    J[5, 6] = dYa_primeOdYLi7
    J[5, 7] = dYa_primeOdYBe7
    J[5, 8] = dYa_primeOdYHe6
    J[5, 9] = dYa_primeOdYLi8
    J[5, 10] = dYa_primeOdYLi6
    J[5, 11] = dYa_primeOdYB8
    J[6, 0] = dYLi7_primeOdYn
    J[6, 1] = dYLi7_primeOdYp
    J[6, 2] = dYLi7_primeOdYd
    J[6, 3] = dYLi7_primeOdYt
    J[6, 4] = dYLi7_primeOdYHe3
    J[6, 5] = dYLi7_primeOdYa
    J[6, 6] = dYLi7_primeOdYLi7
    J[6, 7] = dYLi7_primeOdYBe7
    J[6, 8] = dYLi7_primeOdYHe6
    J[6, 9] = dYLi7_primeOdYLi8
    J[6, 10] = dYLi7_primeOdYLi6
    J[6, 11] = dYLi7_primeOdYB8
    J[7, 0] = dYBe7_primeOdYn
    J[7, 1] = dYBe7_primeOdYp
    J[7, 2] = dYBe7_primeOdYd
    J[7, 3] = dYBe7_primeOdYt
    J[7, 4] = dYBe7_primeOdYHe3
    J[7, 5] = dYBe7_primeOdYa
    J[7, 6] = dYBe7_primeOdYLi7
    J[7, 7] = dYBe7_primeOdYBe7
    J[7, 8] = dYBe7_primeOdYHe6
    J[7, 9] = dYBe7_primeOdYLi8
    J[7, 10] = dYBe7_primeOdYLi6
    J[7, 11] = dYBe7_primeOdYB8
    J[8, 0] = dYHe6_primeOdYn
    J[8, 1] = dYHe6_primeOdYp
    J[8, 2] = dYHe6_primeOdYd
    J[8, 3] = dYHe6_primeOdYt
    J[8, 4] = dYHe6_primeOdYHe3
    J[8, 5] = dYHe6_primeOdYa
    J[8, 6] = dYHe6_primeOdYLi7
    J[8, 7] = dYHe6_primeOdYBe7
    J[8, 8] = dYHe6_primeOdYHe6
    J[8, 9] = dYHe6_primeOdYLi8
    J[8, 10] = dYHe6_primeOdYLi6
    J[8, 11] = dYHe6_primeOdYB8
    J[9, 0] = dYLi8_primeOdYn
    J[9, 1] = dYLi8_primeOdYp
    J[9, 2] = dYLi8_primeOdYd
    J[9, 3] = dYLi8_primeOdYt
    J[9, 4] = dYLi8_primeOdYHe3
    J[9, 5] = dYLi8_primeOdYa
    J[9, 6] = dYLi8_primeOdYLi7
    J[9, 7] = dYLi8_primeOdYBe7
    J[9, 8] = dYLi8_primeOdYHe6
    J[9, 9] = dYLi8_primeOdYLi8
    J[9, 10] = dYLi8_primeOdYLi6
    J[9, 11] = dYLi8_primeOdYB8
    J[10, 0] = dYLi6_primeOdYn
    J[10, 1] = dYLi6_primeOdYp
    J[10, 2] = dYLi6_primeOdYd
    J[10, 3] = dYLi6_primeOdYt
    J[10, 4] = dYLi6_primeOdYHe3
    J[10, 5] = dYLi6_primeOdYa
    J[10, 6] = dYLi6_primeOdYLi7
    J[10, 7] = dYLi6_primeOdYBe7
    J[10, 8] = dYLi6_primeOdYHe6
    J[10, 9] = dYLi6_primeOdYLi8
    J[10, 10] = dYLi6_primeOdYLi6
    J[10, 11] = dYLi6_primeOdYB8
    J[11, 0] = dYB8_primeOdYn
    J[11, 1] = dYB8_primeOdYp
    J[11, 2] = dYB8_primeOdYd
    J[11, 3] = dYB8_primeOdYt
    J[11, 4] = dYB8_primeOdYHe3
    J[11, 5] = dYB8_primeOdYa
    J[11, 6] = dYB8_primeOdYLi7
    J[11, 7] = dYB8_primeOdYBe7
    J[11, 8] = dYB8_primeOdYHe6
    J[11, 9] = dYB8_primeOdYLi8
    J[11, 10] = dYB8_primeOdYLi6
    J[11, 11] = dYB8_primeOdYB8
    return J


def _jacLT_arith(Y, rhoBBN, r):
    Yn1p0, Yn0p1, Yn1p1, Yn2p1, Yn1p2, Yn2p2, Yn4p3, Yn3p4, Yn4p2, Yn5p3, Yn3p3, Yn3p5 = Y

    nTOp_f = r[0]; nTOp_b = r[1]
    B8dBe7He3_f = r[2]; B8dBe7He3_b = r[3]
    B8nBe7d_f = r[4]; B8nBe7d_b = r[5]
    B8nLi6He3_f = r[6]; B8nLi6He3_b = r[7]
    B8naap_f = r[8]; B8naap_b = r[9]
    B8tBe7a_f = r[10]; B8tBe7a_b = r[11]
    B8taaHe3_f = r[12]; B8taaHe3_b = r[13]
    Be7He3aapp_f = r[14]; Be7He3aapp_b = r[15]
    Be7He3ppaa_f = r[16]; Be7He3ppaa_b = r[17]
    Be7daap_f = r[18]; Be7daap_b = r[19]
    Be7nLi7p_f = r[20]; Be7nLi7p_b = r[21]
    Be7naa_f = r[22]; Be7naa_b = r[23]
    Be7pB8g_f = r[24]; Be7pB8g_b = r[25]
    Be7tLi6a_f = r[26]; Be7tLi6a_b = r[27]
    Be7tLi7He3_f = r[28]; Be7tLi7He3_b = r[29]
    Be7taad_f = r[30]; Be7taad_b = r[31]
    Be7taanp_f = r[32]; Be7taanp_b = r[33]
    He3He3app_f = r[34]; He3He3app_b = r[35]
    He3aBe7g_f = r[36]; He3aBe7g_b = r[37]
    He3dap_f = r[38]; He3dap_b = r[39]
    He3nag_f = r[40]; He3nag_b = r[41]
    He3ntp_f = r[42]; He3ntp_b = r[43]
    He3tLi6g_f = r[44]; He3tLi6g_b = r[45]
    He3tad_f = r[46]; He3tad_b = r[47]
    He3tanp_f = r[48]; He3tanp_b = r[49]
    Li6He3Be7d_f = r[50]; Li6He3Be7d_b = r[51]
    Li6He3aap_f = r[52]; Li6He3aap_b = r[53]
    Li6dBe7n_f = r[54]; Li6dBe7n_b = r[55]
    Li6dLi7p_f = r[56]; Li6dLi7p_b = r[57]
    Li6nLi7g_f = r[58]; Li6nLi7g_b = r[59]
    Li6nta_f = r[60]; Li6nta_b = r[61]
    Li6pBe7g_f = r[62]; Li6pBe7g_b = r[63]
    Li6pHe3a_f = r[64]; Li6pHe3a_b = r[65]
    Li6tLi7d_f = r[66]; Li6tLi7d_b = r[67]
    Li6tLi8p_f = r[68]; Li6tLi8p_b = r[69]
    Li6taan_f = r[70]; Li6taan_b = r[71]
    Li7He3Li6a_f = r[72]; Li7He3Li6a_b = r[73]
    Li7He3aad_f = r[74]; Li7He3aad_b = r[75]
    Li7He3aanp_f = r[76]; Li7He3aanp_b = r[77]
    Li7dLi8p_f = r[78]; Li7dLi8p_b = r[79]
    Li7daan_f = r[80]; Li7daan_b = r[81]
    Li7nLi8g_f = r[82]; Li7nLi8g_b = r[83]
    Li7paa_f = r[84]; Li7paa_b = r[85]
    Li7paag_f = r[86]; Li7paag_b = r[87]
    Li7taann_f = r[88]; Li7taann_b = r[89]
    Li8He3Li7a_f = r[90]; Li8He3Li7a_b = r[91]
    Li8He3aat_f = r[92]; Li8He3aat_b = r[93]
    Li8dLi7t_f = r[94]; Li8dLi7t_b = r[95]
    Li8paan_f = r[96]; Li8paan_b = r[97]
    annHe6g_f = r[98]; annHe6g_b = r[99]
    anpLi6g_f = r[100]; anpLi6g_b = r[101]
    daLi6g_f = r[102]; daLi6g_b = r[103]
    ddHe3n_f = r[104]; ddHe3n_b = r[105]
    ddag_f = r[106]; ddag_b = r[107]
    ddtp_f = r[108]; ddtp_b = r[109]
    dntg_f = r[110]; dntg_b = r[111]
    dpHe3g_f = r[112]; dpHe3g_b = r[113]
    npdg_f = r[114]; npdg_b = r[115]
    ppndp_f = r[116]; ppndp_b = r[117]
    taLi7g_f = r[118]; taLi7g_b = r[119]
    tdan_f = r[120]; tdan_b = r[121]
    tpag_f = r[122]; tpag_b = r[123]
    ttann_f = r[124]; ttann_b = r[125]

    # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4}
    # Yn
    dYn_primeOdYn = -2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*annHe6g_f - rhoBBN*rhoBBN*Yn0p1*Yn2p2*anpLi6g_f - rhoBBN*Yn3p5*B8naap_f - rhoBBN*Yn3p5*B8nBe7d_f - rhoBBN*Yn3p5*B8nLi6He3_f - rhoBBN*Yn3p4*Be7naa_f - rhoBBN*Yn3p4*Be7nLi7p_f - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7taanp_b - rhoBBN*Yn1p2*ddHe3n_b - rhoBBN*Yn1p1*dntg_f - rhoBBN*Yn1p2*He3nag_f - rhoBBN*Yn1p2*He3ntp_f - rhoBBN*rhoBBN*Yn0p1*Yn2p2*He3tanp_b - rhoBBN*Yn3p4*Li6dBe7n_b - rhoBBN*Yn3p3*Li6nLi7g_f - rhoBBN*Yn3p3*Li6nta_f - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li6taan_b - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li7daan_b - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Li7He3aanp_b - rhoBBN*Yn4p3*Li7nLi8g_f - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Li7taann_b - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li8paan_b - rhoBBN*Yn0p1*npdg_f - nTOp_f - 0.5*rhoBBN*rhoBBN*Yn0p1*Yn0p1*ppndp_f - rhoBBN*Yn2p2*tdan_b - 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*ttann_b
    dYn_primeOdYp = -rhoBBN*rhoBBN*Yn1p0*Yn2p2*anpLi6g_f + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*B8naap_b + rhoBBN*Yn4p3*Be7nLi7p_b - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Be7taanp_b + rhoBBN*Yn2p1*He3ntp_b - rhoBBN*rhoBBN*Yn1p0*Yn2p2*He3tanp_b - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Li7He3aanp_b + rhoBBN*Yn5p3*Li8paan_f - rhoBBN*Yn1p0*npdg_f + nTOp_b + rhoBBN*Yn1p1*ppndp_b - rhoBBN*rhoBBN*Yn0p1*Yn1p0*ppndp_f
    dYn_primeOdYd = rhoBBN*Yn3p4*B8nBe7d_b + rhoBBN*Yn1p1*ddHe3n_f - rhoBBN*Yn1p0*dntg_f + rhoBBN*Yn3p3*Li6dBe7n_f + rhoBBN*Yn4p3*Li7daan_f + npdg_b + rhoBBN*Yn0p1*ppndp_b + rhoBBN*Yn2p1*tdan_f
    dYn_primeOdYt = rhoBBN*Yn3p4*Be7taanp_f + dntg_b + rhoBBN*Yn0p1*He3ntp_b + rhoBBN*Yn1p2*He3tanp_f + rhoBBN*Yn2p2*Li6nta_b + rhoBBN*Yn3p3*Li6taan_f + 2.*rhoBBN*Yn4p3*Li7taann_f + rhoBBN*Yn1p1*tdan_f + 2.*rhoBBN*Yn2p1*ttann_f
    dYn_primeOdYHe3 = rhoBBN*Yn3p3*B8nLi6He3_b - rhoBBN*Yn1p0*ddHe3n_b - rhoBBN*Yn1p0*He3nag_f - rhoBBN*Yn1p0*He3ntp_f + rhoBBN*Yn2p1*He3tanp_f + rhoBBN*Yn4p3*Li7He3aanp_f
    dYn_primeOdYa = -rhoBBN*rhoBBN*Yn1p0*Yn1p0*annHe6g_f - rhoBBN*rhoBBN*Yn0p1*Yn1p0*anpLi6g_f + rhoBBN*rhoBBN*Yn0p1*Yn2p2*B8naap_b + rhoBBN*Yn2p2*Be7naa_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Be7taanp_b + He3nag_b - rhoBBN*rhoBBN*Yn0p1*Yn1p0*He3tanp_b + rhoBBN*Yn2p1*Li6nta_b - rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li6taan_b - rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li7daan_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Li7He3aanp_b - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn1p0*Yn2p2*Li7taann_b - rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li8paan_b - rhoBBN*Yn1p0*tdan_b - rhoBBN*rhoBBN*Yn1p0*Yn1p0*ttann_b
    dYn_primeOdYLi7 = rhoBBN*Yn0p1*Be7nLi7p_b + Li6nLi7g_b + rhoBBN*Yn1p1*Li7daan_f + rhoBBN*Yn1p2*Li7He3aanp_f - rhoBBN*Yn1p0*Li7nLi8g_f + 2.*rhoBBN*Yn2p1*Li7taann_f
    dYn_primeOdYBe7 = rhoBBN*Yn1p1*B8nBe7d_b - rhoBBN*Yn1p0*Be7naa_f - rhoBBN*Yn1p0*Be7nLi7p_f + rhoBBN*Yn2p1*Be7taanp_f - rhoBBN*Yn1p0*Li6dBe7n_b
    dYn_primeOdYHe6 = 2.*annHe6g_b
    dYn_primeOdYLi8 = Li7nLi8g_b + rhoBBN*Yn0p1*Li8paan_f
    dYn_primeOdYLi6 = anpLi6g_b + rhoBBN*Yn1p2*B8nLi6He3_b + rhoBBN*Yn1p1*Li6dBe7n_f - rhoBBN*Yn1p0*Li6nLi7g_f - rhoBBN*Yn1p0*Li6nta_f + rhoBBN*Yn2p1*Li6taan_f
    dYn_primeOdYB8 = -rhoBBN*Yn1p0*B8naap_f - rhoBBN*Yn1p0*B8nBe7d_f - rhoBBN*Yn1p0*B8nLi6He3_f

    # Yp
    dYp_primeOdYn = -rhoBBN*rhoBBN*Yn0p1*Yn2p2*anpLi6g_f + rhoBBN*Yn3p5*B8naap_f + rhoBBN*Yn3p4*Be7nLi7p_f - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7taanp_b + rhoBBN*Yn1p2*He3ntp_f - rhoBBN*rhoBBN*Yn0p1*Yn2p2*He3tanp_b - 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Li7He3aanp_b + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li8paan_b - rhoBBN*Yn0p1*npdg_f + nTOp_f - 0.5*rhoBBN*rhoBBN*Yn0p1*Yn0p1*ppndp_f
    dYp_primeOdYp = -nTOp_b + rhoBBN*(- npdg_f*Yn1p0 - dpHe3g_f*Yn1p1 - (tpag_f + ddtp_b + He3ntp_b)*Yn2p1 - He3dap_b*Yn2p2 - (Li7paa_f + Be7nLi7p_b)*Yn4p3) + rhoBBN*(-Li7paag_f*Yn4p3) + rhoBBN*(-Li6pBe7g_f*Yn3p3) + rhoBBN*(-Li6pHe3a_f*Yn3p3) + rhoBBN*(-0.5*rhoBBN*B8naap_b*Yn2p2*Yn2p2) + rhoBBN*(-0.5*rhoBBN*Li6He3aap_b*Yn2p2*Yn2p2 - Li6tLi8p_b*Yn5p3) + rhoBBN*(-rhoBBN*rhoBBN*Be7He3ppaa_b*Yn0p1*Yn2p2*Yn2p2 - 2.*rhoBBN*He3He3app_b*Yn0p1*Yn2p2) + rhoBBN*(-rhoBBN*He3tanp_b*Yn1p0*Yn2p2 - 0.5*rhoBBN*rhoBBN*Li7He3aanp_b*Yn1p0*Yn2p2*Yn2p2 - 0.5*rhoBBN*Be7daap_b*Yn2p2*Yn2p2 - 0.5*rhoBBN*rhoBBN*Be7taanp_b*Yn1p0*Yn2p2*Yn2p2 - rhoBBN*rhoBBN*Be7He3aapp_b*Yn0p1*Yn2p2*Yn2p2 - rhoBBN*anpLi6g_f*Yn1p0*Yn2p2 - Li6dLi7p_b*Yn4p3) + rhoBBN*(-Li7dLi8p_b*Yn5p3) + rhoBBN*(-Li8paan_f*Yn5p3) + rhoBBN*(-rhoBBN*ppndp_f*Yn0p1*Yn1p0) + rhoBBN*(ppndp_b*Yn1p1) + rhoBBN*(-Be7pB8g_f*Yn3p4)
    dYp_primeOdYd = rhoBBN*Yn3p4*Be7daap_f + rhoBBN*Yn1p1*ddtp_f - rhoBBN*Yn0p1*dpHe3g_f + rhoBBN*Yn1p2*He3dap_f + rhoBBN*Yn3p3*Li6dLi7p_f + rhoBBN*Yn4p3*Li7dLi8p_f + npdg_b + rhoBBN*Yn0p1*ppndp_b
    dYp_primeOdYt = rhoBBN*Yn3p4*Be7taanp_f - rhoBBN*Yn0p1*ddtp_b - rhoBBN*Yn0p1*He3ntp_b + rhoBBN*Yn1p2*He3tanp_f + rhoBBN*Yn3p3*Li6tLi8p_f - rhoBBN*Yn0p1*tpag_f
    dYp_primeOdYHe3 = 2.*rhoBBN*Yn3p4*Be7He3aapp_f + 2.*rhoBBN*Yn3p4*Be7He3ppaa_f + dpHe3g_b + rhoBBN*Yn1p1*He3dap_f + 2.*rhoBBN*Yn1p2*He3He3app_f + rhoBBN*Yn1p0*He3ntp_f + rhoBBN*Yn2p1*He3tanp_f + rhoBBN*Yn3p3*Li6He3aap_f + rhoBBN*Yn2p2*Li6pHe3a_b + rhoBBN*Yn4p3*Li7He3aanp_f
    dYp_primeOdYa = -rhoBBN*rhoBBN*Yn0p1*Yn1p0*anpLi6g_f - rhoBBN*rhoBBN*Yn0p1*Yn2p2*B8naap_b - rhoBBN*rhoBBN*Yn0p1*Yn2p2*Be7daap_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*Be7He3aapp_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*Be7He3ppaa_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Be7taanp_b - rhoBBN*Yn0p1*He3dap_b - rhoBBN*rhoBBN*Yn0p1*Yn0p1*He3He3app_b - rhoBBN*rhoBBN*Yn0p1*Yn1p0*He3tanp_b - rhoBBN*rhoBBN*Yn0p1*Yn2p2*Li6He3aap_b + rhoBBN*Yn1p2*Li6pHe3a_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Li7He3aanp_b + rhoBBN*Yn2p2*Li7paa_b + rhoBBN*Yn2p2*Li7paag_b + rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li8paan_b + tpag_b
    dYp_primeOdYLi7 = -rhoBBN*Yn0p1*Be7nLi7p_b - rhoBBN*Yn0p1*Li6dLi7p_b + rhoBBN*Yn1p1*Li7dLi8p_f + rhoBBN*Yn1p2*Li7He3aanp_f - rhoBBN*Yn0p1*Li7paa_f - rhoBBN*Yn0p1*Li7paag_f
    dYp_primeOdYBe7 = rhoBBN*Yn1p1*Be7daap_f + 2.*rhoBBN*Yn1p2*Be7He3aapp_f + 2.*rhoBBN*Yn1p2*Be7He3ppaa_f + rhoBBN*Yn1p0*Be7nLi7p_f - rhoBBN*Yn0p1*Be7pB8g_f + rhoBBN*Yn2p1*Be7taanp_f + Li6pBe7g_b
    dYp_primeOdYHe6 = 0.
    dYp_primeOdYLi8 =  -rhoBBN*Yn0p1*Li6tLi8p_b - rhoBBN*Yn0p1*Li7dLi8p_b - rhoBBN*Yn0p1*Li8paan_f
    dYp_primeOdYLi6 = anpLi6g_b + rhoBBN*Yn1p1*Li6dLi7p_f + rhoBBN*Yn1p2*Li6He3aap_f - rhoBBN*Yn0p1*Li6pBe7g_f - rhoBBN*Yn0p1*Li6pHe3a_f + rhoBBN*Yn2p1*Li6tLi8p_f
    dYp_primeOdYB8 = rhoBBN*Yn1p0*B8naap_f + Be7pB8g_b

    # Yd
    dYd_primeOdYn = rhoBBN*Yn3p5*B8nBe7d_f + 2.*rhoBBN*Yn1p2*ddHe3n_b - rhoBBN*Yn1p1*dntg_f + rhoBBN*Yn3p4*Li6dBe7n_b + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li7daan_b + rhoBBN*Yn0p1*npdg_f + 0.5*rhoBBN*rhoBBN*Yn0p1*Yn0p1*ppndp_f + rhoBBN*Yn2p2*tdan_b
    dYd_primeOdYp = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Be7daap_b + 2.*rhoBBN*Yn2p1*ddtp_b - rhoBBN*Yn1p1*dpHe3g_f + rhoBBN*Yn2p2*He3dap_b + rhoBBN*Yn4p3*Li6dLi7p_b + rhoBBN*Yn5p3*Li7dLi8p_b + rhoBBN*Yn1p0*npdg_f - rhoBBN*Yn1p1*ppndp_b + rhoBBN*rhoBBN*Yn0p1*Yn1p0*ppndp_f
    dYd_primeOdYd = -rhoBBN*Yn3p5*B8dBe7He3_f - rhoBBN*Yn3p4*B8nBe7d_b - rhoBBN*Yn3p4*Be7daap_f - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Be7taad_b - rhoBBN*Yn2p2*daLi6g_f - 2.*rhoBBN*Yn1p1*ddag_f - 2.*rhoBBN*Yn1p1*ddHe3n_f - 2.*rhoBBN*Yn1p1*ddtp_f - rhoBBN*Yn1p0*dntg_f - rhoBBN*Yn0p1*dpHe3g_f - rhoBBN*Yn1p2*He3dap_f - rhoBBN*Yn2p2*He3tad_b - rhoBBN*Yn3p3*Li6dBe7n_f - rhoBBN*Yn3p3*Li6dLi7p_f - rhoBBN*Yn3p4*Li6He3Be7d_b - rhoBBN*Yn4p3*Li6tLi7d_b - rhoBBN*Yn4p3*Li7daan_f - rhoBBN*Yn4p3*Li7dLi8p_f - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li7He3aad_b - rhoBBN*Yn5p3*Li8dLi7t_f - npdg_b - rhoBBN*Yn0p1*ppndp_b - rhoBBN*Yn2p1*tdan_f
    dYd_primeOdYt = rhoBBN*Yn3p4*Be7taad_f + 2.*rhoBBN*Yn0p1*ddtp_b + dntg_b + rhoBBN*Yn1p2*He3tad_f + rhoBBN*Yn3p3*Li6tLi7d_f + rhoBBN*Yn4p3*Li8dLi7t_b - rhoBBN*Yn1p1*tdan_f
    dYd_primeOdYHe3 = rhoBBN*Yn3p4*B8dBe7He3_b + 2.*rhoBBN*Yn1p0*ddHe3n_b + dpHe3g_b - rhoBBN*Yn1p1*He3dap_f + rhoBBN*Yn2p1*He3tad_f + rhoBBN*Yn3p3*Li6He3Be7d_f + rhoBBN*Yn4p3*Li7He3aad_f
    dYd_primeOdYa = rhoBBN*rhoBBN*Yn0p1*Yn2p2*Be7daap_b - rhoBBN*rhoBBN*Yn1p1*Yn2p2*Be7taad_b - rhoBBN*Yn1p1*daLi6g_f + 2.*ddag_b + rhoBBN*Yn0p1*He3dap_b - rhoBBN*Yn1p1*He3tad_b + rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li7daan_b - rhoBBN*rhoBBN*Yn1p1*Yn2p2*Li7He3aad_b + rhoBBN*Yn1p0*tdan_b
    dYd_primeOdYLi7 = rhoBBN*Yn0p1*Li6dLi7p_b - rhoBBN*Yn1p1*Li6tLi7d_b - rhoBBN*Yn1p1*Li7daan_f - rhoBBN*Yn1p1*Li7dLi8p_f + rhoBBN*Yn1p2*Li7He3aad_f + rhoBBN*Yn2p1*Li8dLi7t_b
    dYd_primeOdYBe7 = rhoBBN*Yn1p2*B8dBe7He3_b - rhoBBN*Yn1p1*B8nBe7d_b - rhoBBN*Yn1p1*Be7daap_f + rhoBBN*Yn2p1*Be7taad_f + rhoBBN*Yn1p0*Li6dBe7n_b - rhoBBN*Yn1p1*Li6He3Be7d_b
    dYd_primeOdYHe6 = 0.
    dYd_primeOdYLi8 = rhoBBN*Yn0p1*Li7dLi8p_b - rhoBBN*Yn1p1*Li8dLi7t_f
    dYd_primeOdYLi6 = daLi6g_b - rhoBBN*Yn1p1*Li6dBe7n_f - rhoBBN*Yn1p1*Li6dLi7p_f + rhoBBN*Yn1p2*Li6He3Be7d_f + rhoBBN*Yn2p1*Li6tLi7d_f
    dYd_primeOdYB8 = -rhoBBN*Yn1p1*B8dBe7He3_f + rhoBBN*Yn1p0*B8nBe7d_f

    # Yt
    dYt_primeOdYn = 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7taanp_b + rhoBBN*Yn1p1*dntg_f + rhoBBN*Yn1p2*He3ntp_f + rhoBBN*rhoBBN*Yn0p1*Yn2p2*He3tanp_b + rhoBBN*Yn3p3*Li6nta_f + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li6taan_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Li7taann_b + rhoBBN*Yn2p2*tdan_b + 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*ttann_b
    dYt_primeOdYp = 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Be7taanp_b - rhoBBN*Yn2p1*ddtp_b - rhoBBN*Yn2p1*He3ntp_b + rhoBBN*rhoBBN*Yn1p0*Yn2p2*He3tanp_b + rhoBBN*Yn5p3*Li6tLi8p_b - rhoBBN*Yn2p1*tpag_f
    dYt_primeOdYd = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Be7taad_b + rhoBBN*Yn1p1*ddtp_f + rhoBBN*Yn1p0*dntg_f + rhoBBN*Yn2p2*He3tad_b + rhoBBN*Yn4p3*Li6tLi7d_b + rhoBBN*Yn5p3*Li8dLi7t_f - rhoBBN*Yn2p1*tdan_f
    dYt_primeOdYt = -rhoBBN*Yn3p5*B8taaHe3_f - rhoBBN*Yn3p5*B8tBe7a_f - rhoBBN*Yn3p4*Be7taad_f - rhoBBN*Yn3p4*Be7taanp_f - rhoBBN*Yn3p4*Be7tLi6a_f - rhoBBN*Yn3p4*Be7tLi7He3_f - rhoBBN*Yn0p1*ddtp_b - dntg_b - rhoBBN*Yn0p1*He3ntp_b - rhoBBN*Yn1p2*He3tad_f - rhoBBN*Yn1p2*He3tanp_f - rhoBBN*Yn1p2*He3tLi6g_f - rhoBBN*Yn2p2*Li6nta_b - rhoBBN*Yn3p3*Li6taan_f - rhoBBN*Yn3p3*Li6tLi7d_f - rhoBBN*Yn3p3*Li6tLi8p_f - rhoBBN*Yn4p3*Li7taann_f - rhoBBN*Yn4p3*Li8dLi7t_b - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li8He3aat_b - rhoBBN*Yn2p2*taLi7g_f - rhoBBN*Yn0p1*tpag_f - rhoBBN*Yn1p1*tdan_f - 2.*rhoBBN*Yn2p1*ttann_f
    dYt_primeOdYHe3 = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*B8taaHe3_b + rhoBBN*Yn4p3*Be7tLi7He3_b + rhoBBN*Yn1p0*He3ntp_f - rhoBBN*Yn2p1*He3tad_f - rhoBBN*Yn2p1*He3tanp_f - rhoBBN*Yn2p1*He3tLi6g_f + rhoBBN*Yn5p3*Li8He3aat_f
    dYt_primeOdYa = rhoBBN*rhoBBN*Yn1p2*Yn2p2*B8taaHe3_b + rhoBBN*Yn3p4*B8tBe7a_b + rhoBBN*rhoBBN*Yn1p1*Yn2p2*Be7taad_b + rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Be7taanp_b + rhoBBN*Yn3p3*Be7tLi6a_b + rhoBBN*Yn1p1*He3tad_b + rhoBBN*rhoBBN*Yn0p1*Yn1p0*He3tanp_b - rhoBBN*Yn2p1*Li6nta_b + rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li6taan_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn1p0*Yn2p2*Li7taann_b - rhoBBN*rhoBBN*Yn2p1*Yn2p2*Li8He3aat_b - rhoBBN*Yn2p1*taLi7g_f + tpag_b + rhoBBN*Yn1p0*tdan_b + rhoBBN*rhoBBN*Yn1p0*Yn1p0*ttann_b
    dYt_primeOdYLi7 = rhoBBN*Yn1p2*Be7tLi7He3_b + rhoBBN*Yn1p1*Li6tLi7d_b - rhoBBN*Yn2p1*Li7taann_f - rhoBBN*Yn2p1*Li8dLi7t_b + taLi7g_b
    dYt_primeOdYBe7 = rhoBBN*Yn2p2*B8tBe7a_b - rhoBBN*Yn2p1*Be7taad_f - rhoBBN*Yn2p1*Be7taanp_f - rhoBBN*Yn2p1*Be7tLi6a_f - rhoBBN*Yn2p1*Be7tLi7He3_f
    dYt_primeOdYHe6 = 0.
    dYt_primeOdYLi8 = rhoBBN*Yn0p1*Li6tLi8p_b + rhoBBN*Yn1p1*Li8dLi7t_f + rhoBBN*Yn1p2*Li8He3aat_f
    dYt_primeOdYLi6 = rhoBBN*Yn2p2*Be7tLi6a_b + He3tLi6g_b + rhoBBN*Yn1p0*Li6nta_f - rhoBBN*Yn2p1*Li6taan_f - rhoBBN*Yn2p1*Li6tLi7d_f - rhoBBN*Yn2p1*Li6tLi8p_f
    dYt_primeOdYB8 = -rhoBBN*Yn2p1*B8taaHe3_f - rhoBBN*Yn2p1*B8tBe7a_f

    # YHe3
    dYHe3_primeOdYn = rhoBBN*Yn3p5*B8nLi6He3_f - rhoBBN*Yn1p2*ddHe3n_b - rhoBBN*Yn1p2*He3nag_f - rhoBBN*Yn1p2*He3ntp_f + rhoBBN*rhoBBN*Yn0p1*Yn2p2*He3tanp_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Li7He3aanp_b
    dYHe3_primeOdYp = 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7He3aapp_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7He3ppaa_b + rhoBBN*Yn1p1*dpHe3g_f + rhoBBN*Yn2p2*He3dap_b + 2.*rhoBBN*rhoBBN*Yn0p1*Yn2p2*He3He3app_b + rhoBBN*Yn2p1*He3ntp_b + rhoBBN*rhoBBN*Yn1p0*Yn2p2*He3tanp_b + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li6He3aap_b + rhoBBN*Yn3p3*Li6pHe3a_f + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Li7He3aanp_b
    dYHe3_primeOdYd = rhoBBN*Yn3p5*B8dBe7He3_f + rhoBBN*Yn1p1*ddHe3n_f + rhoBBN*Yn0p1*dpHe3g_f - rhoBBN*Yn1p2*He3dap_f + rhoBBN*Yn2p2*He3tad_b + rhoBBN*Yn3p4*Li6He3Be7d_b + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li7He3aad_b
    dYHe3_primeOdYt = rhoBBN*Yn3p5*B8taaHe3_f + rhoBBN*Yn3p4*Be7tLi7He3_f + rhoBBN*Yn0p1*He3ntp_b - rhoBBN*Yn1p2*He3tad_f - rhoBBN*Yn1p2*He3tanp_f - rhoBBN*Yn1p2*He3tLi6g_f + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li8He3aat_b
    dYHe3_primeOdYHe3 = -rhoBBN*Yn3p4*B8dBe7He3_b - rhoBBN*Yn3p3*B8nLi6He3_b - 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*B8taaHe3_b - rhoBBN*Yn3p4*Be7He3aapp_f - rhoBBN*Yn3p4*Be7He3ppaa_f - rhoBBN*Yn4p3*Be7tLi7He3_b - rhoBBN*Yn1p0*ddHe3n_b - dpHe3g_b - rhoBBN*Yn2p2*He3aBe7g_f - rhoBBN*Yn1p1*He3dap_f - 2.*rhoBBN*Yn1p2*He3He3app_f - rhoBBN*Yn1p0*He3nag_f - rhoBBN*Yn1p0*He3ntp_f - rhoBBN*Yn2p1*He3tad_f - rhoBBN*Yn2p1*He3tanp_f - rhoBBN*Yn2p1*He3tLi6g_f - rhoBBN*Yn3p3*Li6He3aap_f - rhoBBN*Yn3p3*Li6He3Be7d_f - rhoBBN*Yn2p2*Li6pHe3a_b - rhoBBN*Yn4p3*Li7He3aad_f - rhoBBN*Yn4p3*Li7He3aanp_f - rhoBBN*Yn4p3*Li7He3Li6a_f - rhoBBN*Yn5p3*Li8He3aat_f - rhoBBN*Yn5p3*Li8He3Li7a_f
    dYHe3_primeOdYa = -rhoBBN*rhoBBN*Yn1p2*Yn2p2*B8taaHe3_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*Be7He3aapp_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*Be7He3ppaa_b - rhoBBN*Yn1p2*He3aBe7g_f + rhoBBN*Yn0p1*He3dap_b + rhoBBN*rhoBBN*Yn0p1*Yn0p1*He3He3app_b + He3nag_b + rhoBBN*Yn1p1*He3tad_b + rhoBBN*rhoBBN*Yn0p1*Yn1p0*He3tanp_b + rhoBBN*rhoBBN*Yn0p1*Yn2p2*Li6He3aap_b - rhoBBN*Yn1p2*Li6pHe3a_b + rhoBBN*rhoBBN*Yn1p1*Yn2p2*Li7He3aad_b + rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Li7He3aanp_b + rhoBBN*Yn3p3*Li7He3Li6a_b + rhoBBN*rhoBBN*Yn2p1*Yn2p2*Li8He3aat_b + rhoBBN*Yn4p3*Li8He3Li7a_b
    dYHe3_primeOdYLi7 = -rhoBBN*Yn1p2*Be7tLi7He3_b - rhoBBN*Yn1p2*Li7He3aad_f - rhoBBN*Yn1p2*Li7He3aanp_f - rhoBBN*Yn1p2*Li7He3Li6a_f + rhoBBN*Yn2p2*Li8He3Li7a_b
    dYHe3_primeOdYBe7 = -rhoBBN*Yn1p2*B8dBe7He3_b - rhoBBN*Yn1p2*Be7He3aapp_f - rhoBBN*Yn1p2*Be7He3ppaa_f + rhoBBN*Yn2p1*Be7tLi7He3_f + He3aBe7g_b + rhoBBN*Yn1p1*Li6He3Be7d_b
    dYHe3_primeOdYHe6 = 0.
    dYHe3_primeOdYLi8 = -rhoBBN*Yn1p2*Li8He3aat_f - rhoBBN*Yn1p2*Li8He3Li7a_f
    dYHe3_primeOdYLi6 = -rhoBBN*Yn1p2*B8nLi6He3_b + He3tLi6g_b - rhoBBN*Yn1p2*Li6He3aap_f - rhoBBN*Yn1p2*Li6He3Be7d_f + rhoBBN*Yn0p1*Li6pHe3a_f + rhoBBN*Yn2p2*Li7He3Li6a_b
    dYHe3_primeOdYB8 = rhoBBN*Yn1p1*B8dBe7He3_f + rhoBBN*Yn1p0*B8nLi6He3_f + rhoBBN*Yn2p1*B8taaHe3_f

    # Ya
    dYa_primeOdYn = -rhoBBN*rhoBBN*Yn1p0*Yn2p2*annHe6g_f - rhoBBN*rhoBBN*Yn0p1*Yn2p2*anpLi6g_f + 2.*rhoBBN*Yn3p5*B8naap_f + 2.*rhoBBN*Yn3p4*Be7naa_f - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7taanp_b + rhoBBN*Yn1p2*He3nag_f - rhoBBN*rhoBBN*Yn0p1*Yn2p2*He3tanp_b + rhoBBN*Yn3p3*Li6nta_f - rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li6taan_b - rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li7daan_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Li7He3aanp_b - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Li7taann_b - rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li8paan_b - rhoBBN*Yn2p2*tdan_b - rhoBBN*rhoBBN*Yn1p0*Yn2p2*ttann_b
    dYa_primeOdYp = -rhoBBN*rhoBBN*Yn1p0*Yn2p2*anpLi6g_f - rhoBBN*rhoBBN*Yn2p2*Yn2p2*B8naap_b - rhoBBN*rhoBBN*Yn2p2*Yn2p2*Be7daap_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7He3aapp_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7He3ppaa_b - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Be7taanp_b - rhoBBN*Yn2p2*He3dap_b - rhoBBN*rhoBBN*Yn0p1*Yn2p2*He3He3app_b - rhoBBN*rhoBBN*Yn1p0*Yn2p2*He3tanp_b - rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li6He3aap_b + rhoBBN*Yn3p3*Li6pHe3a_f - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Li7He3aanp_b + 2.*rhoBBN*Yn4p3*Li7paa_f + 2.*rhoBBN*Yn4p3*Li7paag_f + 2.*rhoBBN*Yn5p3*Li8paan_f + rhoBBN*Yn2p1*tpag_f
    dYa_primeOdYd = 2.*rhoBBN*Yn3p4*Be7daap_f - rhoBBN*rhoBBN*Yn2p2*Yn2p2*Be7taad_b - rhoBBN*Yn2p2*daLi6g_f + rhoBBN*Yn1p1*ddag_f + rhoBBN*Yn1p2*He3dap_f - rhoBBN*Yn2p2*He3tad_b + 2.*rhoBBN*Yn4p3*Li7daan_f - rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li7He3aad_b + rhoBBN*Yn2p1*tdan_f
    dYa_primeOdYt = 2.*rhoBBN*Yn3p5*B8taaHe3_f + rhoBBN*Yn3p5*B8tBe7a_f + 2.*rhoBBN*Yn3p4*Be7taad_f + 2.*rhoBBN*Yn3p4*Be7taanp_f + rhoBBN*Yn3p4*Be7tLi6a_f + rhoBBN*Yn1p2*He3tad_f + rhoBBN*Yn1p2*He3tanp_f - rhoBBN*Yn2p2*Li6nta_b + 2.*rhoBBN*Yn3p3*Li6taan_f + 2.*rhoBBN*Yn4p3*Li7taann_f - rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li8He3aat_b - rhoBBN*Yn2p2*taLi7g_f + rhoBBN*Yn0p1*tpag_f + rhoBBN*Yn1p1*tdan_f + rhoBBN*Yn2p1*ttann_f
    dYa_primeOdYHe3 = -rhoBBN*rhoBBN*Yn2p2*Yn2p2*B8taaHe3_b + 2.*rhoBBN*Yn3p4*Be7He3aapp_f + 2.*rhoBBN*Yn3p4*Be7He3ppaa_f - rhoBBN*Yn2p2*He3aBe7g_f + rhoBBN*Yn1p1*He3dap_f + rhoBBN*Yn1p2*He3He3app_f + rhoBBN*Yn1p0*He3nag_f + rhoBBN*Yn2p1*He3tad_f + rhoBBN*Yn2p1*He3tanp_f + 2.*rhoBBN*Yn3p3*Li6He3aap_f - rhoBBN*Yn2p2*Li6pHe3a_b + 2.*rhoBBN*Yn4p3*Li7He3aad_f + 2.*rhoBBN*Yn4p3*Li7He3aanp_f + rhoBBN*Yn4p3*Li7He3Li6a_f + 2.*rhoBBN*Yn5p3*Li8He3aat_f + rhoBBN*Yn5p3*Li8He3Li7a_f
    dYa_primeOdYa = -0.5*rhoBBN*rhoBBN*Yn1p0*Yn1p0*annHe6g_f - rhoBBN*rhoBBN*Yn0p1*Yn1p0*anpLi6g_f - 2.*rhoBBN*rhoBBN*Yn0p1*Yn2p2*B8naap_b - 2.*rhoBBN*rhoBBN*Yn1p2*Yn2p2*B8taaHe3_b - rhoBBN*Yn3p4*B8tBe7a_b - 2.*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Be7daap_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*Be7He3aapp_b - rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*Be7He3ppaa_b - 2.*rhoBBN*Yn2p2*Be7naa_b - 2.*rhoBBN*rhoBBN*Yn1p1*Yn2p2*Be7taad_b - 2.*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Be7taanp_b - rhoBBN*Yn3p3*Be7tLi6a_b - rhoBBN*Yn1p1*daLi6g_f - ddag_b - rhoBBN*Yn1p2*He3aBe7g_f - rhoBBN*Yn0p1*He3dap_b - 0.5*rhoBBN*rhoBBN*Yn0p1*Yn0p1*He3He3app_b - He3nag_b - rhoBBN*Yn1p1*He3tad_b - rhoBBN*rhoBBN*Yn0p1*Yn1p0*He3tanp_b - 2.*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Li6He3aap_b - rhoBBN*Yn2p1*Li6nta_b - rhoBBN*Yn1p2*Li6pHe3a_b - 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li6taan_b - 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li7daan_b - 2.*rhoBBN*rhoBBN*Yn1p1*Yn2p2*Li7He3aad_b - 2.*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Li7He3aanp_b - rhoBBN*Yn3p3*Li7He3Li6a_b - 2.*rhoBBN*Yn2p2*Li7paa_b - 2.*rhoBBN*Yn2p2*Li7paag_b - rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn1p0*Yn2p2*Li7taann_b - 2.*rhoBBN*rhoBBN*Yn2p1*Yn2p2*Li8He3aat_b - rhoBBN*Yn4p3*Li8He3Li7a_b - 2.*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li8paan_b - rhoBBN*Yn2p1*taLi7g_f - tpag_b - rhoBBN*Yn1p0*tdan_b - 0.5*rhoBBN*rhoBBN*Yn1p0*Yn1p0*ttann_b
    dYa_primeOdYLi7 = 2.*rhoBBN*Yn1p1*Li7daan_f + 2.*rhoBBN*Yn1p2*Li7He3aad_f + 2.*rhoBBN*Yn1p2*Li7He3aanp_f + rhoBBN*Yn1p2*Li7He3Li6a_f + 2.*rhoBBN*Yn0p1*Li7paa_f + 2.*rhoBBN*Yn0p1*Li7paag_f + 2.*rhoBBN*Yn2p1*Li7taann_f - rhoBBN*Yn2p2*Li8He3Li7a_b + taLi7g_b
    dYa_primeOdYBe7 = -rhoBBN*Yn2p2*B8tBe7a_b + 2.*rhoBBN*Yn1p1*Be7daap_f + 2.*rhoBBN*Yn1p2*Be7He3aapp_f + 2.*rhoBBN*Yn1p2*Be7He3ppaa_f + 2.*rhoBBN*Yn1p0*Be7naa_f + 2.*rhoBBN*Yn2p1*Be7taad_f + 2.*rhoBBN*Yn2p1*Be7taanp_f + rhoBBN*Yn2p1*Be7tLi6a_f + He3aBe7g_b
    dYa_primeOdYHe6 = annHe6g_b
    dYa_primeOdYLi8 = 2.*rhoBBN*Yn1p2*Li8He3aat_f + rhoBBN*Yn1p2*Li8He3Li7a_f + 2.*rhoBBN*Yn0p1*Li8paan_f
    dYa_primeOdYLi6 = anpLi6g_b - rhoBBN*Yn2p2*Be7tLi6a_b + daLi6g_b + 2.*rhoBBN*Yn1p2*Li6He3aap_f + rhoBBN*Yn1p0*Li6nta_f + rhoBBN*Yn0p1*Li6pHe3a_f + 2.*rhoBBN*Yn2p1*Li6taan_f - rhoBBN*Yn2p2*Li7He3Li6a_b
    dYa_primeOdYB8 = 2.*rhoBBN*Yn1p0*B8naap_f + 2.*rhoBBN*Yn2p1*B8taaHe3_f + rhoBBN*Yn2p1*B8tBe7a_f

    # YLi7
    dYLi7_primeOdYn = rhoBBN*Yn3p4*Be7nLi7p_f + rhoBBN*Yn3p3*Li6nLi7g_f + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li7daan_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Li7He3aanp_b - rhoBBN*Yn4p3*Li7nLi8g_f + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Li7taann_b
    dYLi7_primeOdYp = -rhoBBN*Yn4p3*Be7nLi7p_b - rhoBBN*Yn4p3*Li6dLi7p_b + rhoBBN*Yn5p3*Li7dLi8p_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Li7He3aanp_b - rhoBBN*Yn4p3*Li7paa_f - rhoBBN*Yn4p3*Li7paag_f
    dYLi7_primeOdYd = rhoBBN*Yn3p3*Li6dLi7p_f - rhoBBN*Yn4p3*Li6tLi7d_b - rhoBBN*Yn4p3*Li7daan_f - rhoBBN*Yn4p3*Li7dLi8p_f + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li7He3aad_b + rhoBBN*Yn5p3*Li8dLi7t_f
    dYLi7_primeOdYt = rhoBBN*Yn3p4*Be7tLi7He3_f + rhoBBN*Yn3p3*Li6tLi7d_f - rhoBBN*Yn4p3*Li7taann_f - rhoBBN*Yn4p3*Li8dLi7t_b + rhoBBN*Yn2p2*taLi7g_f
    dYLi7_primeOdYHe3 = -rhoBBN*Yn4p3*Be7tLi7He3_b - rhoBBN*Yn4p3*Li7He3aad_f - rhoBBN*Yn4p3*Li7He3aanp_f - rhoBBN*Yn4p3*Li7He3Li6a_f + rhoBBN*Yn5p3*Li8He3Li7a_f
    dYLi7_primeOdYa = rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li7daan_b + rhoBBN*rhoBBN*Yn1p1*Yn2p2*Li7He3aad_b + rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Li7He3aanp_b + rhoBBN*Yn3p3*Li7He3Li6a_b + rhoBBN*Yn2p2*Li7paa_b + rhoBBN*Yn2p2*Li7paag_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn1p0*Yn2p2*Li7taann_b - rhoBBN*Yn4p3*Li8He3Li7a_b + rhoBBN*Yn2p1*taLi7g_f
    dYLi7_primeOdYLi7 = -rhoBBN*Yn0p1*Be7nLi7p_b - rhoBBN*Yn1p2*Be7tLi7He3_b - rhoBBN*Yn0p1*Li6dLi7p_b - Li6nLi7g_b - rhoBBN*Yn1p1*Li6tLi7d_b - rhoBBN*Yn1p1*Li7daan_f - rhoBBN*Yn1p1*Li7dLi8p_f - rhoBBN*Yn1p2*Li7He3aad_f - rhoBBN*Yn1p2*Li7He3aanp_f - rhoBBN*Yn1p2*Li7He3Li6a_f - rhoBBN*Yn1p0*Li7nLi8g_f - rhoBBN*Yn0p1*Li7paa_f - rhoBBN*Yn0p1*Li7paag_f - rhoBBN*Yn2p1*Li7taann_f - rhoBBN*Yn2p1*Li8dLi7t_b - rhoBBN*Yn2p2*Li8He3Li7a_b - taLi7g_b
    dYLi7_primeOdYBe7 = rhoBBN*Yn1p0*Be7nLi7p_f + rhoBBN*Yn2p1*Be7tLi7He3_f
    dYLi7_primeOdYHe6 = 0.
    dYLi7_primeOdYLi8 = rhoBBN*Yn0p1*Li7dLi8p_b + Li7nLi8g_b + rhoBBN*Yn1p1*Li8dLi7t_f + rhoBBN*Yn1p2*Li8He3Li7a_f
    dYLi7_primeOdYLi6 = rhoBBN*Yn1p1*Li6dLi7p_f + rhoBBN*Yn1p0*Li6nLi7g_f + rhoBBN*Yn2p1*Li6tLi7d_f + rhoBBN*Yn2p2*Li7He3Li6a_b
    dYLi7_primeOdYB8 = 0.

    # YBe7
    dYBe7_primeOdYn = rhoBBN*Yn3p5*B8nBe7d_f - rhoBBN*Yn3p4*Be7naa_f - rhoBBN*Yn3p4*Be7nLi7p_f + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7taanp_b - rhoBBN*Yn3p4*Li6dBe7n_b
    dYBe7_primeOdYp = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Be7daap_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7He3aapp_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn2p2*Yn2p2*Be7He3ppaa_b + rhoBBN*Yn4p3*Be7nLi7p_b - rhoBBN*Yn3p4*Be7pB8g_f + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn1p0*Yn2p2*Yn2p2*Be7taanp_b + rhoBBN*Yn3p3*Li6pBe7g_f
    dYBe7_primeOdYd = rhoBBN*Yn3p5*B8dBe7He3_f - rhoBBN*Yn3p4*B8nBe7d_b - rhoBBN*Yn3p4*Be7daap_f + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Be7taad_b + rhoBBN*Yn3p3*Li6dBe7n_f - rhoBBN*Yn3p4*Li6He3Be7d_b
    dYBe7_primeOdYt = rhoBBN*Yn3p5*B8tBe7a_f - rhoBBN*Yn3p4*Be7taad_f - rhoBBN*Yn3p4*Be7taanp_f - rhoBBN*Yn3p4*Be7tLi6a_f - rhoBBN*Yn3p4*Be7tLi7He3_f
    dYBe7_primeOdYHe3 = -rhoBBN*Yn3p4*B8dBe7He3_b - rhoBBN*Yn3p4*Be7He3aapp_f - rhoBBN*Yn3p4*Be7He3ppaa_f + rhoBBN*Yn4p3*Be7tLi7He3_b + rhoBBN*Yn2p2*He3aBe7g_f + rhoBBN*Yn3p3*Li6He3Be7d_f
    dYBe7_primeOdYa = -rhoBBN*Yn3p4*B8tBe7a_b + rhoBBN*rhoBBN*Yn0p1*Yn2p2*Be7daap_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*Be7He3aapp_b + 0.5*rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn0p1*Yn2p2*Be7He3ppaa_b + rhoBBN*Yn2p2*Be7naa_b + rhoBBN*rhoBBN*Yn1p1*Yn2p2*Be7taad_b + rhoBBN*rhoBBN*rhoBBN*Yn0p1*Yn1p0*Yn2p2*Be7taanp_b + rhoBBN*Yn3p3*Be7tLi6a_b + rhoBBN*Yn1p2*He3aBe7g_f
    dYBe7_primeOdYLi7 = rhoBBN*Yn0p1*Be7nLi7p_b + rhoBBN*Yn1p2*Be7tLi7He3_b
    dYBe7_primeOdYBe7 = -rhoBBN*Yn1p2*B8dBe7He3_b - rhoBBN*Yn1p1*B8nBe7d_b - rhoBBN*Yn2p2*B8tBe7a_b - rhoBBN*Yn1p1*Be7daap_f - rhoBBN*Yn1p2*Be7He3aapp_f - rhoBBN*Yn1p2*Be7He3ppaa_f - rhoBBN*Yn1p0*Be7naa_f - rhoBBN*Yn1p0*Be7nLi7p_f - rhoBBN*Yn0p1*Be7pB8g_f - rhoBBN*Yn2p1*Be7taad_f - rhoBBN*Yn2p1*Be7taanp_f - rhoBBN*Yn2p1*Be7tLi6a_f - rhoBBN*Yn2p1*Be7tLi7He3_f - He3aBe7g_b - rhoBBN*Yn1p0*Li6dBe7n_b - rhoBBN*Yn1p1*Li6He3Be7d_b - Li6pBe7g_b
    dYBe7_primeOdYHe6 = 0.
    dYBe7_primeOdYLi8 = 0.
    dYBe7_primeOdYLi6 = rhoBBN*Yn2p2*Be7tLi6a_b + rhoBBN*Yn1p1*Li6dBe7n_f + rhoBBN*Yn1p2*Li6He3Be7d_f + rhoBBN*Yn0p1*Li6pBe7g_f
    dYBe7_primeOdYB8 = rhoBBN*Yn1p1*B8dBe7He3_f + rhoBBN*Yn1p0*B8nBe7d_f + rhoBBN*Yn2p1*B8tBe7a_f + Be7pB8g_b

    # YHe6
    dYHe6_primeOdYn = rhoBBN*rhoBBN*Yn1p0*Yn2p2*annHe6g_f
    dYHe6_primeOdYp = 0.
    dYHe6_primeOdYd = 0.
    dYHe6_primeOdYt = 0.
    dYHe6_primeOdYHe3 = 0.
    dYHe6_primeOdYa = 0.5*rhoBBN*rhoBBN*Yn1p0*Yn1p0*annHe6g_f
    dYHe6_primeOdYLi7 = 0.
    dYHe6_primeOdYBe7 = 0.
    dYHe6_primeOdYHe6 = -annHe6g_b
    dYHe6_primeOdYLi8 = 0.
    dYHe6_primeOdYLi6 = 0.
    dYHe6_primeOdYB8 = 0.

    # YLi8
    dYLi8_primeOdYn = rhoBBN*Yn4p3*Li7nLi8g_f + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li8paan_b
    dYLi8_primeOdYp = -rhoBBN*Yn5p3*Li6tLi8p_b - rhoBBN*Yn5p3*Li7dLi8p_b - rhoBBN*Yn5p3*Li8paan_f
    dYLi8_primeOdYd = rhoBBN*Yn4p3*Li7dLi8p_f - rhoBBN*Yn5p3*Li8dLi7t_f
    dYLi8_primeOdYt = rhoBBN*Yn3p3*Li6tLi8p_f + rhoBBN*Yn4p3*Li8dLi7t_b + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li8He3aat_b
    dYLi8_primeOdYHe3 = -rhoBBN*Yn5p3*Li8He3aat_f - rhoBBN*Yn5p3*Li8He3Li7a_f
    dYLi8_primeOdYa = rhoBBN*rhoBBN*Yn2p1*Yn2p2*Li8He3aat_b + rhoBBN*Yn4p3*Li8He3Li7a_b + rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li8paan_b
    dYLi8_primeOdYLi7 = rhoBBN*Yn1p1*Li7dLi8p_f + rhoBBN*Yn1p0*Li7nLi8g_f + rhoBBN*Yn2p1*Li8dLi7t_b + rhoBBN*Yn2p2*Li8He3Li7a_b
    dYLi8_primeOdYBe7 = 0.
    dYLi8_primeOdYHe6 = 0.
    dYLi8_primeOdYLi8 = -rhoBBN*Yn0p1*Li6tLi8p_b - rhoBBN*Yn0p1*Li7dLi8p_b - Li7nLi8g_b - rhoBBN*Yn1p1*Li8dLi7t_f - rhoBBN*Yn1p2*Li8He3aat_f - rhoBBN*Yn1p2*Li8He3Li7a_f - rhoBBN*Yn0p1*Li8paan_f
    dYLi8_primeOdYLi6 = rhoBBN*Yn2p1*Li6tLi8p_f
    dYLi8_primeOdYB8 = 0.

    # YLi6
    dYLi6_primeOdYn = rhoBBN*rhoBBN*Yn0p1*Yn2p2*anpLi6g_f + rhoBBN*Yn3p5*B8nLi6He3_f + rhoBBN*Yn3p4*Li6dBe7n_b - rhoBBN*Yn3p3*Li6nLi7g_f - rhoBBN*Yn3p3*Li6nta_f + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li6taan_b
    dYLi6_primeOdYp = rhoBBN*rhoBBN*Yn1p0*Yn2p2*anpLi6g_f + rhoBBN*Yn4p3*Li6dLi7p_b + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*Li6He3aap_b - rhoBBN*Yn3p3*Li6pBe7g_f - rhoBBN*Yn3p3*Li6pHe3a_f + rhoBBN*Yn5p3*Li6tLi8p_b
    dYLi6_primeOdYd = rhoBBN*Yn2p2*daLi6g_f - rhoBBN*Yn3p3*Li6dBe7n_f - rhoBBN*Yn3p3*Li6dLi7p_f + rhoBBN*Yn3p4*Li6He3Be7d_b + rhoBBN*Yn4p3*Li6tLi7d_b
    dYLi6_primeOdYt = rhoBBN*Yn3p4*Be7tLi6a_f + rhoBBN*Yn1p2*He3tLi6g_f + rhoBBN*Yn2p2*Li6nta_b - rhoBBN*Yn3p3*Li6taan_f - rhoBBN*Yn3p3*Li6tLi7d_f - rhoBBN*Yn3p3*Li6tLi8p_f
    dYLi6_primeOdYHe3 = -rhoBBN*Yn3p3*B8nLi6He3_b + rhoBBN*Yn2p1*He3tLi6g_f - rhoBBN*Yn3p3*Li6He3aap_f - rhoBBN*Yn3p3*Li6He3Be7d_f + rhoBBN*Yn2p2*Li6pHe3a_b + rhoBBN*Yn4p3*Li7He3Li6a_f
    dYLi6_primeOdYa = rhoBBN*rhoBBN*Yn0p1*Yn1p0*anpLi6g_f - rhoBBN*Yn3p3*Be7tLi6a_b + rhoBBN*Yn1p1*daLi6g_f + rhoBBN*rhoBBN*Yn0p1*Yn2p2*Li6He3aap_b + rhoBBN*Yn2p1*Li6nta_b + rhoBBN*Yn1p2*Li6pHe3a_b + rhoBBN*rhoBBN*Yn1p0*Yn2p2*Li6taan_b - rhoBBN*Yn3p3*Li7He3Li6a_b
    dYLi6_primeOdYLi7 = rhoBBN*Yn0p1*Li6dLi7p_b + Li6nLi7g_b + rhoBBN*Yn1p1*Li6tLi7d_b + rhoBBN*Yn1p2*Li7He3Li6a_f
    dYLi6_primeOdYBe7 = rhoBBN*Yn2p1*Be7tLi6a_f + rhoBBN*Yn1p0*Li6dBe7n_b + rhoBBN*Yn1p1*Li6He3Be7d_b + Li6pBe7g_b
    dYLi6_primeOdYHe6 = 0.
    dYLi6_primeOdYLi8 = rhoBBN*Yn0p1*Li6tLi8p_b
    dYLi6_primeOdYLi6 = -anpLi6g_b - rhoBBN*Yn1p2*B8nLi6He3_b - rhoBBN*Yn2p2*Be7tLi6a_b - daLi6g_b - He3tLi6g_b - rhoBBN*Yn1p1*Li6dBe7n_f - rhoBBN*Yn1p1*Li6dLi7p_f - rhoBBN*Yn1p2*Li6He3aap_f - rhoBBN*Yn1p2*Li6He3Be7d_f - rhoBBN*Yn1p0*Li6nLi7g_f - rhoBBN*Yn1p0*Li6nta_f - rhoBBN*Yn0p1*Li6pBe7g_f - rhoBBN*Yn0p1*Li6pHe3a_f - rhoBBN*Yn2p1*Li6taan_f - rhoBBN*Yn2p1*Li6tLi7d_f - rhoBBN*Yn2p1*Li6tLi8p_f - rhoBBN*Yn2p2*Li7He3Li6a_b
    dYLi6_primeOdYB8 = rhoBBN*Yn1p0*B8nLi6He3_f

    # {Yn -> Yn1p0, Yp -> Yn0p1, Yd -> Yn1p1, Yt -> Yn2p1, YHe3 -> Yn1p2, Ya -> Yn2p2, YLi7 -> Yn4p3, YBe7 -> Yn3p4, YHe6 -> Yn4p2, Li8 -> Yn5p3, Li6 -> Yn3p3, B8 -> Yn3p5}
    # YB8
    dYB8_primeOdYn = -rhoBBN*Yn3p5*B8naap_f - rhoBBN*Yn3p5*B8nBe7d_f - rhoBBN*Yn3p5*B8nLi6He3_f
    dYB8_primeOdYp = 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*B8naap_b + rhoBBN*Yn3p4*Be7pB8g_f
    dYB8_primeOdYd = -rhoBBN*Yn3p5*B8dBe7He3_f + rhoBBN*Yn3p4*B8nBe7d_b
    dYB8_primeOdYt = -rhoBBN*Yn3p5*B8taaHe3_f - rhoBBN*Yn3p5*B8tBe7a_f
    dYB8_primeOdYHe3 = rhoBBN*Yn3p4*B8dBe7He3_b + rhoBBN*Yn3p3*B8nLi6He3_b + 0.5*rhoBBN*rhoBBN*Yn2p2*Yn2p2*B8taaHe3_b
    dYB8_primeOdYa = rhoBBN*rhoBBN*Yn0p1*Yn2p2*B8naap_b + rhoBBN*rhoBBN*Yn1p2*Yn2p2*B8taaHe3_b + rhoBBN*Yn3p4*B8tBe7a_b
    dYB8_primeOdYLi7 = 0.
    dYB8_primeOdYBe7 = rhoBBN*Yn1p2*B8dBe7He3_b + rhoBBN*Yn1p1*B8nBe7d_b + rhoBBN*Yn2p2*B8tBe7a_b + rhoBBN*Yn0p1*Be7pB8g_f
    dYB8_primeOdYHe6 = 0.
    dYB8_primeOdYLi8 = 0.
    dYB8_primeOdYLi6 = rhoBBN*Yn1p2*B8nLi6He3_b
    dYB8_primeOdYB8 = -rhoBBN*Yn1p1*B8dBe7He3_f - rhoBBN*Yn1p0*B8naap_f - rhoBBN*Yn1p0*B8nBe7d_f - rhoBBN*Yn1p0*B8nLi6He3_f - rhoBBN*Yn2p1*B8taaHe3_f - rhoBBN*Yn2p1*B8tBe7a_f - Be7pB8g_b




    J = np.empty((12, 12))
    J[0, 0] = dYn_primeOdYn
    J[0, 1] = dYn_primeOdYp
    J[0, 2] = dYn_primeOdYd
    J[0, 3] = dYn_primeOdYt
    J[0, 4] = dYn_primeOdYHe3
    J[0, 5] = dYn_primeOdYa
    J[0, 6] = dYn_primeOdYLi7
    J[0, 7] = dYn_primeOdYBe7
    J[0, 8] = dYn_primeOdYHe6
    J[0, 9] = dYn_primeOdYLi8
    J[0, 10] = dYn_primeOdYLi6
    J[0, 11] = dYn_primeOdYB8
    J[1, 0] = dYp_primeOdYn
    J[1, 1] = dYp_primeOdYp
    J[1, 2] = dYp_primeOdYd
    J[1, 3] = dYp_primeOdYt
    J[1, 4] = dYp_primeOdYHe3
    J[1, 5] = dYp_primeOdYa
    J[1, 6] = dYp_primeOdYLi7
    J[1, 7] = dYp_primeOdYBe7
    J[1, 8] = dYp_primeOdYHe6
    J[1, 9] = dYp_primeOdYLi8
    J[1, 10] = dYp_primeOdYLi6
    J[1, 11] = dYp_primeOdYB8
    J[2, 0] = dYd_primeOdYn
    J[2, 1] = dYd_primeOdYp
    J[2, 2] = dYd_primeOdYd
    J[2, 3] = dYd_primeOdYt
    J[2, 4] = dYd_primeOdYHe3
    J[2, 5] = dYd_primeOdYa
    J[2, 6] = dYd_primeOdYLi7
    J[2, 7] = dYd_primeOdYBe7
    J[2, 8] = dYd_primeOdYHe6
    J[2, 9] = dYd_primeOdYLi8
    J[2, 10] = dYd_primeOdYLi6
    J[2, 11] = dYd_primeOdYB8
    J[3, 0] = dYt_primeOdYn
    J[3, 1] = dYt_primeOdYp
    J[3, 2] = dYt_primeOdYd
    J[3, 3] = dYt_primeOdYt
    J[3, 4] = dYt_primeOdYHe3
    J[3, 5] = dYt_primeOdYa
    J[3, 6] = dYt_primeOdYLi7
    J[3, 7] = dYt_primeOdYBe7
    J[3, 8] = dYt_primeOdYHe6
    J[3, 9] = dYt_primeOdYLi8
    J[3, 10] = dYt_primeOdYLi6
    J[3, 11] = dYt_primeOdYB8
    J[4, 0] = dYHe3_primeOdYn
    J[4, 1] = dYHe3_primeOdYp
    J[4, 2] = dYHe3_primeOdYd
    J[4, 3] = dYHe3_primeOdYt
    J[4, 4] = dYHe3_primeOdYHe3
    J[4, 5] = dYHe3_primeOdYa
    J[4, 6] = dYHe3_primeOdYLi7
    J[4, 7] = dYHe3_primeOdYBe7
    J[4, 8] = dYHe3_primeOdYHe6
    J[4, 9] = dYHe3_primeOdYLi8
    J[4, 10] = dYHe3_primeOdYLi6
    J[4, 11] = dYHe3_primeOdYB8
    J[5, 0] = dYa_primeOdYn
    J[5, 1] = dYa_primeOdYp
    J[5, 2] = dYa_primeOdYd
    J[5, 3] = dYa_primeOdYt
    J[5, 4] = dYa_primeOdYHe3
    J[5, 5] = dYa_primeOdYa
    J[5, 6] = dYa_primeOdYLi7
    J[5, 7] = dYa_primeOdYBe7
    J[5, 8] = dYa_primeOdYHe6
    J[5, 9] = dYa_primeOdYLi8
    J[5, 10] = dYa_primeOdYLi6
    J[5, 11] = dYa_primeOdYB8
    J[6, 0] = dYLi7_primeOdYn
    J[6, 1] = dYLi7_primeOdYp
    J[6, 2] = dYLi7_primeOdYd
    J[6, 3] = dYLi7_primeOdYt
    J[6, 4] = dYLi7_primeOdYHe3
    J[6, 5] = dYLi7_primeOdYa
    J[6, 6] = dYLi7_primeOdYLi7
    J[6, 7] = dYLi7_primeOdYBe7
    J[6, 8] = dYLi7_primeOdYHe6
    J[6, 9] = dYLi7_primeOdYLi8
    J[6, 10] = dYLi7_primeOdYLi6
    J[6, 11] = dYLi7_primeOdYB8
    J[7, 0] = dYBe7_primeOdYn
    J[7, 1] = dYBe7_primeOdYp
    J[7, 2] = dYBe7_primeOdYd
    J[7, 3] = dYBe7_primeOdYt
    J[7, 4] = dYBe7_primeOdYHe3
    J[7, 5] = dYBe7_primeOdYa
    J[7, 6] = dYBe7_primeOdYLi7
    J[7, 7] = dYBe7_primeOdYBe7
    J[7, 8] = dYBe7_primeOdYHe6
    J[7, 9] = dYBe7_primeOdYLi8
    J[7, 10] = dYBe7_primeOdYLi6
    J[7, 11] = dYBe7_primeOdYB8
    J[8, 0] = dYHe6_primeOdYn
    J[8, 1] = dYHe6_primeOdYp
    J[8, 2] = dYHe6_primeOdYd
    J[8, 3] = dYHe6_primeOdYt
    J[8, 4] = dYHe6_primeOdYHe3
    J[8, 5] = dYHe6_primeOdYa
    J[8, 6] = dYHe6_primeOdYLi7
    J[8, 7] = dYHe6_primeOdYBe7
    J[8, 8] = dYHe6_primeOdYHe6
    J[8, 9] = dYHe6_primeOdYLi8
    J[8, 10] = dYHe6_primeOdYLi6
    J[8, 11] = dYHe6_primeOdYB8
    J[9, 0] = dYLi8_primeOdYn
    J[9, 1] = dYLi8_primeOdYp
    J[9, 2] = dYLi8_primeOdYd
    J[9, 3] = dYLi8_primeOdYt
    J[9, 4] = dYLi8_primeOdYHe3
    J[9, 5] = dYLi8_primeOdYa
    J[9, 6] = dYLi8_primeOdYLi7
    J[9, 7] = dYLi8_primeOdYBe7
    J[9, 8] = dYLi8_primeOdYHe6
    J[9, 9] = dYLi8_primeOdYLi8
    J[9, 10] = dYLi8_primeOdYLi6
    J[9, 11] = dYLi8_primeOdYB8
    J[10, 0] = dYLi6_primeOdYn
    J[10, 1] = dYLi6_primeOdYp
    J[10, 2] = dYLi6_primeOdYd
    J[10, 3] = dYLi6_primeOdYt
    J[10, 4] = dYLi6_primeOdYHe3
    J[10, 5] = dYLi6_primeOdYa
    J[10, 6] = dYLi6_primeOdYLi7
    J[10, 7] = dYLi6_primeOdYBe7
    J[10, 8] = dYLi6_primeOdYHe6
    J[10, 9] = dYLi6_primeOdYLi8
    J[10, 10] = dYLi6_primeOdYLi6
    J[10, 11] = dYLi6_primeOdYB8
    J[11, 0] = dYB8_primeOdYn
    J[11, 1] = dYB8_primeOdYp
    J[11, 2] = dYB8_primeOdYd
    J[11, 3] = dYB8_primeOdYt
    J[11, 4] = dYB8_primeOdYHe3
    J[11, 5] = dYB8_primeOdYa
    J[11, 6] = dYB8_primeOdYLi7
    J[11, 7] = dYB8_primeOdYBe7
    J[11, 8] = dYB8_primeOdYHe6
    J[11, 9] = dYB8_primeOdYLi8
    J[11, 10] = dYB8_primeOdYLi6
    J[11, 11] = dYB8_primeOdYB8
    return J


class UpdateNuclearRates:
    """
    Build interpolated forward and backward rates for nuclear reactions.

    Parameters
    ----------
    nuclear_data : NuclearData
        Rate tables loaded from disk.
    cfg : PyPRConfig
        Configuration (flags, NP shifts).  When ``cfg.smallnet_flag`` is True,
        only the 12 key reactions are loaded; otherwise all 63 reactions.
    """

    def __init__(self, nuclear_data, cfg):

        nd = nuclear_data
        NP = cfg.NP_nuclear_flag
        nreac = 12 if cfg.smallnet_flag else 63
        if cfg.verbose_flag:
            print(f"[rates] Building rate interpolants for {nreac} reactions.")

        def _rate(median, expsigma, p, NP_delta, NP_flag):
            mu = median * np.exp(p * np.log(expsigma))
            if NP_flag:
                mu += NP_delta * median
            return mu

        def _sp(T9, mu, kind='linear'):
            return interp1d(T9, mu, bounds_error=False,
                            fill_value="extrapolate", kind=kind)

        # ------------------------------------------------------------------
        # 12 key reactions (linear splines) — always loaded
        # ------------------------------------------------------------------
        for _rxn in _KEY12_REACTIONS:
            setattr(self, f'{_rxn}_spline',
                    _sp(getattr(nd, f'{_rxn}_T9'),
                        _rate(getattr(nd, f'{_rxn}_median'),
                              getattr(nd, f'{_rxn}_expsigma'),
                              getattr(cfg, f'p_{_rxn}'),
                              getattr(cfg, f'NP_delta_{_rxn}'),
                              NP)))

        if not cfg.smallnet_flag:
            # --------------------------------------------------------------
            # 29 extra linear reactions
            # --------------------------------------------------------------
            for _rxn in _EXTRA29_LINEAR:
                setattr(self, f'{_rxn}_spline',
                        _sp(getattr(nd, f'{_rxn}_T9'),
                            _rate(getattr(nd, f'{_rxn}_median'),
                                  getattr(nd, f'{_rxn}_expsigma'),
                                  getattr(cfg, f'p_{_rxn}'),
                                  getattr(cfg, f'NP_delta_{_rxn}'),
                                  NP)))

            # --------------------------------------------------------------
            # 22 quadratic reactions
            # --------------------------------------------------------------
            for _rxn in _EXTRA22_QUADRATIC:
                setattr(self, f'{_rxn}_spline',
                        _sp(getattr(nd, f'{_rxn}_T9'),
                            _rate(getattr(nd, f'{_rxn}_median'),
                                  getattr(nd, f'{_rxn}_expsigma'),
                                  getattr(cfg, f'p_{_rxn}'),
                                  getattr(cfg, f'NP_delta_{_rxn}'),
                                  NP),
                            kind='quadratic'))

        # ------------------------------------------------------------------
        # Store detailed-balance coefficients for all loaded reactions
        # ------------------------------------------------------------------
        loaded_rxns = _KEY12_REACTIONS if cfg.smallnet_flag else _ALL_REACTIONS
        for _rxn in loaded_rxns:
            setattr(self, f'_alpha_{_rxn}', getattr(nd, f'alpha_{_rxn}'))
            setattr(self, f'_beta_{_rxn}',  getattr(nd, f'beta_{_rxn}'))
            setattr(self, f'_gamma_{_rxn}', getattr(nd, f'gamma_{_rxn}'))

        self._rhs_rbuf = np.empty(26)
        self._rhsMT_rbuf = np.empty(36)
        self._rhsLT_rbuf = np.empty(126)
        self._jac_rbuf   = np.empty(26)
        self._jacMT_rbuf = np.empty(36)
        self._jacLT_rbuf = np.empty(126)
        _setup_nuclear_rhs_impls(cfg.numba_flag)

    def rhs(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        r = self._rhs_rbuf
        r[0] = nTOp_frwrd(T_t); r[1] = nTOp_bkwrd(T_t)
        r[2] = self.npdg_frwrd(T_t); r[3] = self.npdg_bkwrd(T_t)
        r[4] = self.dpHe3g_frwrd(T_t); r[5] = self.dpHe3g_bkwrd(T_t)
        r[6] = self.ddHe3n_frwrd(T_t); r[7] = self.ddHe3n_bkwrd(T_t)
        r[8] = self.ddtp_frwrd(T_t); r[9] = self.ddtp_bkwrd(T_t)
        r[10] = self.tpag_frwrd(T_t); r[11] = self.tpag_bkwrd(T_t)
        r[12] = self.tdan_frwrd(T_t); r[13] = self.tdan_bkwrd(T_t)
        r[14] = self.taLi7g_frwrd(T_t); r[15] = self.taLi7g_bkwrd(T_t)
        r[16] = self.He3ntp_frwrd(T_t); r[17] = self.He3ntp_bkwrd(T_t)
        r[18] = self.He3dap_frwrd(T_t); r[19] = self.He3dap_bkwrd(T_t)
        r[20] = self.He3aBe7g_frwrd(T_t); r[21] = self.He3aBe7g_bkwrd(T_t)
        r[22] = self.Be7nLi7p_frwrd(T_t); r[23] = self.Be7nLi7p_bkwrd(T_t)
        r[24] = self.Li7paa_frwrd(T_t); r[25] = self.Li7paa_bkwrd(T_t)
        return _rhs_impl(Y, rhoBBN, r)

    def Jacobian(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        r = self._jac_rbuf
        r[0] = nTOp_frwrd(T_t); r[1] = nTOp_bkwrd(T_t)
        r[2] = self.npdg_frwrd(T_t); r[3] = self.npdg_bkwrd(T_t)
        r[4] = self.dpHe3g_frwrd(T_t); r[5] = self.dpHe3g_bkwrd(T_t)
        r[6] = self.ddHe3n_frwrd(T_t); r[7] = self.ddHe3n_bkwrd(T_t)
        r[8] = self.ddtp_frwrd(T_t); r[9] = self.ddtp_bkwrd(T_t)
        r[10] = self.tpag_frwrd(T_t); r[11] = self.tpag_bkwrd(T_t)
        r[12] = self.tdan_frwrd(T_t); r[13] = self.tdan_bkwrd(T_t)
        r[14] = self.taLi7g_frwrd(T_t); r[15] = self.taLi7g_bkwrd(T_t)
        r[16] = self.He3ntp_frwrd(T_t); r[17] = self.He3ntp_bkwrd(T_t)
        r[18] = self.He3dap_frwrd(T_t); r[19] = self.He3dap_bkwrd(T_t)
        r[20] = self.He3aBe7g_frwrd(T_t); r[21] = self.He3aBe7g_bkwrd(T_t)
        r[22] = self.Be7nLi7p_frwrd(T_t); r[23] = self.Be7nLi7p_bkwrd(T_t)
        r[24] = self.Li7paa_frwrd(T_t); r[25] = self.Li7paa_bkwrd(T_t)
        return _jac_impl(Y, rhoBBN, r)

    def rhsMT(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        r = self._rhsMT_rbuf
        r[0] = nTOp_frwrd(T_t); r[1] = nTOp_bkwrd(T_t)
        r[2] = self.Be7daap_frwrd(T_t); r[3] = self.Be7daap_bkwrd(T_t)
        r[4] = self.Be7nLi7p_frwrd(T_t); r[5] = self.Be7nLi7p_bkwrd(T_t)
        r[6] = self.Be7naa_frwrd(T_t); r[7] = self.Be7naa_bkwrd(T_t)
        r[8] = self.He3aBe7g_frwrd(T_t); r[9] = self.He3aBe7g_bkwrd(T_t)
        r[10] = self.He3dap_frwrd(T_t); r[11] = self.He3dap_bkwrd(T_t)
        r[12] = self.He3ntp_frwrd(T_t); r[13] = self.He3ntp_bkwrd(T_t)
        r[14] = self.Li6pBe7g_frwrd(T_t); r[15] = self.Li6pBe7g_bkwrd(T_t)
        r[16] = self.Li7paa_frwrd(T_t); r[17] = self.Li7paa_bkwrd(T_t)
        r[18] = self.Li7paag_frwrd(T_t); r[19] = self.Li7paag_bkwrd(T_t)
        r[20] = self.daLi6g_frwrd(T_t); r[21] = self.daLi6g_bkwrd(T_t)
        r[22] = self.ddHe3n_frwrd(T_t); r[23] = self.ddHe3n_bkwrd(T_t)
        r[24] = self.ddtp_frwrd(T_t); r[25] = self.ddtp_bkwrd(T_t)
        r[26] = self.dpHe3g_frwrd(T_t); r[27] = self.dpHe3g_bkwrd(T_t)
        r[28] = self.npdg_frwrd(T_t); r[29] = self.npdg_bkwrd(T_t)
        r[30] = self.taLi7g_frwrd(T_t); r[31] = self.taLi7g_bkwrd(T_t)
        r[32] = self.tdan_frwrd(T_t); r[33] = self.tdan_bkwrd(T_t)
        r[34] = self.tpag_frwrd(T_t); r[35] = self.tpag_bkwrd(T_t)
        return _rhsMT_impl(Y, rhoBBN, r)

    def JacobianMT(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        r = self._jacMT_rbuf
        r[0] = nTOp_frwrd(T_t); r[1] = nTOp_bkwrd(T_t)
        r[2] = self.Be7daap_frwrd(T_t); r[3] = self.Be7daap_bkwrd(T_t)
        r[4] = self.Be7nLi7p_frwrd(T_t); r[5] = self.Be7nLi7p_bkwrd(T_t)
        r[6] = self.Be7naa_frwrd(T_t); r[7] = self.Be7naa_bkwrd(T_t)
        r[8] = self.He3aBe7g_frwrd(T_t); r[9] = self.He3aBe7g_bkwrd(T_t)
        r[10] = self.He3dap_frwrd(T_t); r[11] = self.He3dap_bkwrd(T_t)
        r[12] = self.He3ntp_frwrd(T_t); r[13] = self.He3ntp_bkwrd(T_t)
        r[14] = self.Li6pBe7g_frwrd(T_t); r[15] = self.Li6pBe7g_bkwrd(T_t)
        r[16] = self.Li7paa_frwrd(T_t); r[17] = self.Li7paa_bkwrd(T_t)
        r[18] = self.Li7paag_frwrd(T_t); r[19] = self.Li7paag_bkwrd(T_t)
        r[20] = self.daLi6g_frwrd(T_t); r[21] = self.daLi6g_bkwrd(T_t)
        r[22] = self.ddHe3n_frwrd(T_t); r[23] = self.ddHe3n_bkwrd(T_t)
        r[24] = self.ddtp_frwrd(T_t); r[25] = self.ddtp_bkwrd(T_t)
        r[26] = self.dpHe3g_frwrd(T_t); r[27] = self.dpHe3g_bkwrd(T_t)
        r[28] = self.npdg_frwrd(T_t); r[29] = self.npdg_bkwrd(T_t)
        r[30] = self.taLi7g_frwrd(T_t); r[31] = self.taLi7g_bkwrd(T_t)
        r[32] = self.tdan_frwrd(T_t); r[33] = self.tdan_bkwrd(T_t)
        r[34] = self.tpag_frwrd(T_t); r[35] = self.tpag_bkwrd(T_t)
        return _jacMT_impl(Y, rhoBBN, r)

    def rhsLT(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        r = self._rhsLT_rbuf
        r[0] = nTOp_frwrd(T_t); r[1] = nTOp_bkwrd(T_t)
        r[2] = self.B8dBe7He3_frwrd(T_t); r[3] = self.B8dBe7He3_bkwrd(T_t)
        r[4] = self.B8nBe7d_frwrd(T_t); r[5] = self.B8nBe7d_bkwrd(T_t)
        r[6] = self.B8nLi6He3_frwrd(T_t); r[7] = self.B8nLi6He3_bkwrd(T_t)
        r[8] = self.B8naap_frwrd(T_t); r[9] = self.B8naap_bkwrd(T_t)
        r[10] = self.B8tBe7a_frwrd(T_t); r[11] = self.B8tBe7a_bkwrd(T_t)
        r[12] = self.B8taaHe3_frwrd(T_t); r[13] = self.B8taaHe3_bkwrd(T_t)
        r[14] = self.Be7He3aapp_frwrd(T_t); r[15] = self.Be7He3aapp_bkwrd(T_t)
        r[16] = self.Be7He3ppaa_frwrd(T_t); r[17] = self.Be7He3ppaa_bkwrd(T_t)
        r[18] = self.Be7daap_frwrd(T_t); r[19] = self.Be7daap_bkwrd(T_t)
        r[20] = self.Be7nLi7p_frwrd(T_t); r[21] = self.Be7nLi7p_bkwrd(T_t)
        r[22] = self.Be7naa_frwrd(T_t); r[23] = self.Be7naa_bkwrd(T_t)
        r[24] = self.Be7pB8g_frwrd(T_t); r[25] = self.Be7pB8g_bkwrd(T_t)
        r[26] = self.Be7tLi6a_frwrd(T_t); r[27] = self.Be7tLi6a_bkwrd(T_t)
        r[28] = self.Be7tLi7He3_frwrd(T_t); r[29] = self.Be7tLi7He3_bkwrd(T_t)
        r[30] = self.Be7taad_frwrd(T_t); r[31] = self.Be7taad_bkwrd(T_t)
        r[32] = self.Be7taanp_frwrd(T_t); r[33] = self.Be7taanp_bkwrd(T_t)
        r[34] = self.He3He3app_frwrd(T_t); r[35] = self.He3He3app_bkwrd(T_t)
        r[36] = self.He3aBe7g_frwrd(T_t); r[37] = self.He3aBe7g_bkwrd(T_t)
        r[38] = self.He3dap_frwrd(T_t); r[39] = self.He3dap_bkwrd(T_t)
        r[40] = self.He3nag_frwrd(T_t); r[41] = self.He3nag_bkwrd(T_t)
        r[42] = self.He3ntp_frwrd(T_t); r[43] = self.He3ntp_bkwrd(T_t)
        r[44] = self.He3tLi6g_frwrd(T_t); r[45] = self.He3tLi6g_bkwrd(T_t)
        r[46] = self.He3tad_frwrd(T_t); r[47] = self.He3tad_bkwrd(T_t)
        r[48] = self.He3tanp_frwrd(T_t); r[49] = self.He3tanp_bkwrd(T_t)
        r[50] = self.Li6He3Be7d_frwrd(T_t); r[51] = self.Li6He3Be7d_bkwrd(T_t)
        r[52] = self.Li6He3aap_frwrd(T_t); r[53] = self.Li6He3aap_bkwrd(T_t)
        r[54] = self.Li6dBe7n_frwrd(T_t); r[55] = self.Li6dBe7n_bkwrd(T_t)
        r[56] = self.Li6dLi7p_frwrd(T_t); r[57] = self.Li6dLi7p_bkwrd(T_t)
        r[58] = self.Li6nLi7g_frwrd(T_t); r[59] = self.Li6nLi7g_bkwrd(T_t)
        r[60] = self.Li6nta_frwrd(T_t); r[61] = self.Li6nta_bkwrd(T_t)
        r[62] = self.Li6pBe7g_frwrd(T_t); r[63] = self.Li6pBe7g_bkwrd(T_t)
        r[64] = self.Li6pHe3a_frwrd(T_t); r[65] = self.Li6pHe3a_bkwrd(T_t)
        r[66] = self.Li6tLi7d_frwrd(T_t); r[67] = self.Li6tLi7d_bkwrd(T_t)
        r[68] = self.Li6tLi8p_frwrd(T_t); r[69] = self.Li6tLi8p_bkwrd(T_t)
        r[70] = self.Li6taan_frwrd(T_t); r[71] = self.Li6taan_bkwrd(T_t)
        r[72] = self.Li7He3Li6a_frwrd(T_t); r[73] = self.Li7He3Li6a_bkwrd(T_t)
        r[74] = self.Li7He3aad_frwrd(T_t); r[75] = self.Li7He3aad_bkwrd(T_t)
        r[76] = self.Li7He3aanp_frwrd(T_t); r[77] = self.Li7He3aanp_bkwrd(T_t)
        r[78] = self.Li7dLi8p_frwrd(T_t); r[79] = self.Li7dLi8p_bkwrd(T_t)
        r[80] = self.Li7daan_frwrd(T_t); r[81] = self.Li7daan_bkwrd(T_t)
        r[82] = self.Li7nLi8g_frwrd(T_t); r[83] = self.Li7nLi8g_bkwrd(T_t)
        r[84] = self.Li7paa_frwrd(T_t); r[85] = self.Li7paa_bkwrd(T_t)
        r[86] = self.Li7paag_frwrd(T_t); r[87] = self.Li7paag_bkwrd(T_t)
        r[88] = self.Li7taann_frwrd(T_t); r[89] = self.Li7taann_bkwrd(T_t)
        r[90] = self.Li8He3Li7a_frwrd(T_t); r[91] = self.Li8He3Li7a_bkwrd(T_t)
        r[92] = self.Li8He3aat_frwrd(T_t); r[93] = self.Li8He3aat_bkwrd(T_t)
        r[94] = self.Li8dLi7t_frwrd(T_t); r[95] = self.Li8dLi7t_bkwrd(T_t)
        r[96] = self.Li8paan_frwrd(T_t); r[97] = self.Li8paan_bkwrd(T_t)
        r[98] = self.annHe6g_frwrd(T_t); r[99] = self.annHe6g_bkwrd(T_t)
        r[100] = self.anpLi6g_frwrd(T_t); r[101] = self.anpLi6g_bkwrd(T_t)
        r[102] = self.daLi6g_frwrd(T_t); r[103] = self.daLi6g_bkwrd(T_t)
        r[104] = self.ddHe3n_frwrd(T_t); r[105] = self.ddHe3n_bkwrd(T_t)
        r[106] = self.ddag_frwrd(T_t); r[107] = self.ddag_bkwrd(T_t)
        r[108] = self.ddtp_frwrd(T_t); r[109] = self.ddtp_bkwrd(T_t)
        r[110] = self.dntg_frwrd(T_t); r[111] = self.dntg_bkwrd(T_t)
        r[112] = self.dpHe3g_frwrd(T_t); r[113] = self.dpHe3g_bkwrd(T_t)
        r[114] = self.npdg_frwrd(T_t); r[115] = self.npdg_bkwrd(T_t)
        r[116] = self.ppndp_frwrd(T_t); r[117] = self.ppndp_bkwrd(T_t)
        r[118] = self.taLi7g_frwrd(T_t); r[119] = self.taLi7g_bkwrd(T_t)
        r[120] = self.tdan_frwrd(T_t); r[121] = self.tdan_bkwrd(T_t)
        r[122] = self.tpag_frwrd(T_t); r[123] = self.tpag_bkwrd(T_t)
        r[124] = self.ttann_frwrd(T_t); r[125] = self.ttann_bkwrd(T_t)
        return _rhsLT_impl(Y, rhoBBN, r)

    def JacobianLT(self, Y, T_t, rhoBBN, nTOp_frwrd, nTOp_bkwrd):
        r = self._jacLT_rbuf
        r[0] = nTOp_frwrd(T_t); r[1] = nTOp_bkwrd(T_t)
        r[2] = self.B8dBe7He3_frwrd(T_t); r[3] = self.B8dBe7He3_bkwrd(T_t)
        r[4] = self.B8nBe7d_frwrd(T_t); r[5] = self.B8nBe7d_bkwrd(T_t)
        r[6] = self.B8nLi6He3_frwrd(T_t); r[7] = self.B8nLi6He3_bkwrd(T_t)
        r[8] = self.B8naap_frwrd(T_t); r[9] = self.B8naap_bkwrd(T_t)
        r[10] = self.B8tBe7a_frwrd(T_t); r[11] = self.B8tBe7a_bkwrd(T_t)
        r[12] = self.B8taaHe3_frwrd(T_t); r[13] = self.B8taaHe3_bkwrd(T_t)
        r[14] = self.Be7He3aapp_frwrd(T_t); r[15] = self.Be7He3aapp_bkwrd(T_t)
        r[16] = self.Be7He3ppaa_frwrd(T_t); r[17] = self.Be7He3ppaa_bkwrd(T_t)
        r[18] = self.Be7daap_frwrd(T_t); r[19] = self.Be7daap_bkwrd(T_t)
        r[20] = self.Be7nLi7p_frwrd(T_t); r[21] = self.Be7nLi7p_bkwrd(T_t)
        r[22] = self.Be7naa_frwrd(T_t); r[23] = self.Be7naa_bkwrd(T_t)
        r[24] = self.Be7pB8g_frwrd(T_t); r[25] = self.Be7pB8g_bkwrd(T_t)
        r[26] = self.Be7tLi6a_frwrd(T_t); r[27] = self.Be7tLi6a_bkwrd(T_t)
        r[28] = self.Be7tLi7He3_frwrd(T_t); r[29] = self.Be7tLi7He3_bkwrd(T_t)
        r[30] = self.Be7taad_frwrd(T_t); r[31] = self.Be7taad_bkwrd(T_t)
        r[32] = self.Be7taanp_frwrd(T_t); r[33] = self.Be7taanp_bkwrd(T_t)
        r[34] = self.He3He3app_frwrd(T_t); r[35] = self.He3He3app_bkwrd(T_t)
        r[36] = self.He3aBe7g_frwrd(T_t); r[37] = self.He3aBe7g_bkwrd(T_t)
        r[38] = self.He3dap_frwrd(T_t); r[39] = self.He3dap_bkwrd(T_t)
        r[40] = self.He3nag_frwrd(T_t); r[41] = self.He3nag_bkwrd(T_t)
        r[42] = self.He3ntp_frwrd(T_t); r[43] = self.He3ntp_bkwrd(T_t)
        r[44] = self.He3tLi6g_frwrd(T_t); r[45] = self.He3tLi6g_bkwrd(T_t)
        r[46] = self.He3tad_frwrd(T_t); r[47] = self.He3tad_bkwrd(T_t)
        r[48] = self.He3tanp_frwrd(T_t); r[49] = self.He3tanp_bkwrd(T_t)
        r[50] = self.Li6He3Be7d_frwrd(T_t); r[51] = self.Li6He3Be7d_bkwrd(T_t)
        r[52] = self.Li6He3aap_frwrd(T_t); r[53] = self.Li6He3aap_bkwrd(T_t)
        r[54] = self.Li6dBe7n_frwrd(T_t); r[55] = self.Li6dBe7n_bkwrd(T_t)
        r[56] = self.Li6dLi7p_frwrd(T_t); r[57] = self.Li6dLi7p_bkwrd(T_t)
        r[58] = self.Li6nLi7g_frwrd(T_t); r[59] = self.Li6nLi7g_bkwrd(T_t)
        r[60] = self.Li6nta_frwrd(T_t); r[61] = self.Li6nta_bkwrd(T_t)
        r[62] = self.Li6pBe7g_frwrd(T_t); r[63] = self.Li6pBe7g_bkwrd(T_t)
        r[64] = self.Li6pHe3a_frwrd(T_t); r[65] = self.Li6pHe3a_bkwrd(T_t)
        r[66] = self.Li6tLi7d_frwrd(T_t); r[67] = self.Li6tLi7d_bkwrd(T_t)
        r[68] = self.Li6tLi8p_frwrd(T_t); r[69] = self.Li6tLi8p_bkwrd(T_t)
        r[70] = self.Li6taan_frwrd(T_t); r[71] = self.Li6taan_bkwrd(T_t)
        r[72] = self.Li7He3Li6a_frwrd(T_t); r[73] = self.Li7He3Li6a_bkwrd(T_t)
        r[74] = self.Li7He3aad_frwrd(T_t); r[75] = self.Li7He3aad_bkwrd(T_t)
        r[76] = self.Li7He3aanp_frwrd(T_t); r[77] = self.Li7He3aanp_bkwrd(T_t)
        r[78] = self.Li7dLi8p_frwrd(T_t); r[79] = self.Li7dLi8p_bkwrd(T_t)
        r[80] = self.Li7daan_frwrd(T_t); r[81] = self.Li7daan_bkwrd(T_t)
        r[82] = self.Li7nLi8g_frwrd(T_t); r[83] = self.Li7nLi8g_bkwrd(T_t)
        r[84] = self.Li7paa_frwrd(T_t); r[85] = self.Li7paa_bkwrd(T_t)
        r[86] = self.Li7paag_frwrd(T_t); r[87] = self.Li7paag_bkwrd(T_t)
        r[88] = self.Li7taann_frwrd(T_t); r[89] = self.Li7taann_bkwrd(T_t)
        r[90] = self.Li8He3Li7a_frwrd(T_t); r[91] = self.Li8He3Li7a_bkwrd(T_t)
        r[92] = self.Li8He3aat_frwrd(T_t); r[93] = self.Li8He3aat_bkwrd(T_t)
        r[94] = self.Li8dLi7t_frwrd(T_t); r[95] = self.Li8dLi7t_bkwrd(T_t)
        r[96] = self.Li8paan_frwrd(T_t); r[97] = self.Li8paan_bkwrd(T_t)
        r[98] = self.annHe6g_frwrd(T_t); r[99] = self.annHe6g_bkwrd(T_t)
        r[100] = self.anpLi6g_frwrd(T_t); r[101] = self.anpLi6g_bkwrd(T_t)
        r[102] = self.daLi6g_frwrd(T_t); r[103] = self.daLi6g_bkwrd(T_t)
        r[104] = self.ddHe3n_frwrd(T_t); r[105] = self.ddHe3n_bkwrd(T_t)
        r[106] = self.ddag_frwrd(T_t); r[107] = self.ddag_bkwrd(T_t)
        r[108] = self.ddtp_frwrd(T_t); r[109] = self.ddtp_bkwrd(T_t)
        r[110] = self.dntg_frwrd(T_t); r[111] = self.dntg_bkwrd(T_t)
        r[112] = self.dpHe3g_frwrd(T_t); r[113] = self.dpHe3g_bkwrd(T_t)
        r[114] = self.npdg_frwrd(T_t); r[115] = self.npdg_bkwrd(T_t)
        r[116] = self.ppndp_frwrd(T_t); r[117] = self.ppndp_bkwrd(T_t)
        r[118] = self.taLi7g_frwrd(T_t); r[119] = self.taLi7g_bkwrd(T_t)
        r[120] = self.tdan_frwrd(T_t); r[121] = self.tdan_bkwrd(T_t)
        r[122] = self.tpag_frwrd(T_t); r[123] = self.tpag_bkwrd(T_t)
        r[124] = self.ttann_frwrd(T_t); r[125] = self.ttann_bkwrd(T_t)
        return _jacLT_impl(Y, rhoBBN, r)

# ---------------------------------------------------------------------------
# Dynamically add forward / backward rate methods for all reactions.
# This eliminates 126 explicit method definitions and fixes the duplicate-
# definition bug present in the old net63 file.
# ---------------------------------------------------------------------------

def _make_frwrd(rxn):
    def frwrd(self, T):
        return getattr(self, f'{rxn}_spline')(T * 1e-9)
    frwrd.__name__ = f'{rxn}_frwrd'
    return frwrd


def _make_bkwrd(rxn):
    def bkwrd(self, T):
        T9 = T * 1e-9
        return (getattr(self, f'_alpha_{rxn}') * T9 ** getattr(self, f'_beta_{rxn}')
                * np.exp(getattr(self, f'_gamma_{rxn}') / T9)
                * getattr(self, f'{rxn}_spline')(T9))
    bkwrd.__name__ = f'{rxn}_bkwrd'
    return bkwrd


for _rxn in _ALL_REACTIONS:
    setattr(UpdateNuclearRates, f'{_rxn}_frwrd', _make_frwrd(_rxn))
    setattr(UpdateNuclearRates, f'{_rxn}_bkwrd', _make_bkwrd(_rxn))
