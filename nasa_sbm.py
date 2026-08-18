"""
nasa_sbm.py — NASA Standard Break-Up Model (Johnson et al., 2001)
"""

import numpy as np
import math
from sgp4.api import Satrec, WGS72, jday


def generate_fragments(m_target, m_projectile, v_impact,
                       parent_pos, parent_vel,
                       jd_epoch, fr_epoch,
                       min_size=0.05, max_fragments=None,
                       seed=42):
    """
    Generate debris fragments from a collision.
    """
    rng = np.random.default_rng(seed)
    mu = 398600.4418

    # Catastrophic if specific energy > 40 J/g
    E_specific = (m_projectile * (v_impact * 1000)**2) / (2 * m_target * 1000)  # J/g
    catastrophic = E_specific >= 40.0
    print(f"Specific energy: {E_specific:.1f} J/g -> {'Catastrophic' if catastrophic else 'Non-catastrophic'}")
    
    if catastrophic:
        M = m_target + m_projectile  # total mass, kg
    else:
        M = m_projectile * (v_impact * 1000)**2 / (1000 * 1000)  # "equivalent mass"
    
    # n_f = 0.1 * M^0.75 * lc^(-1.71) for collision
    lc = min_size  # minimum characteristic length in meters
    n_fragments = int(0.1 * M**0.75 * lc**(-1.71))
    
    if max_fragments and n_fragments > max_fragments:
        print(f"Capping fragments from {n_fragments} to {max_fragments}")
        n_fragments = max_fragments
    print(f"Generating {n_fragments} fragments (lc >= {min_size} m)")
    
    # Cumulative: N(>lc) = 0.1 * M^0.75 * lc^(-1.71)
    # Sample from power law: lc ~ lc_min * U^(-1/1.71)
    u = rng.uniform(0, 1, n_fragments)
    sizes = min_size * (1 - u)**(-1/1.71)  # characteristic length in meters
    sizes = np.clip(sizes, min_size, 10.0)  # cap at 10m (no fragment bigger than parent)
    
    # Log-normal distribution, parameters depend on size
    log_lc = np.log10(sizes)
    am_ratios = np.zeros(n_fragments)
    
    for i in range(n_fragments):
        if log_lc[i] < -1.75:
            mu_am = -0.3
            sigma_am = 0.2
        elif log_lc[i] < -1.25:
            mu_am = -0.3 - 1.4 * (log_lc[i] + 1.75)
            sigma_am = 0.2 + 0.1 * (log_lc[i] + 1.75)
        else:
            mu_am = -0.6 - 2.0 * (log_lc[i] + 1.25)
            sigma_am = 0.3
        
        log_am = rng.normal(mu_am, max(sigma_am, 0.01))
        am_ratios[i] = 10**log_am  # m^2/kg
    
    # Log-normal in velocity, depends on A/M
    log_am = np.log10(am_ratios)
    mu_v = 0.2 * log_am + 1.85
    sigma_v = 0.4
    
    delta_vs = 10**rng.normal(mu_v, sigma_v)  # m/s
    delta_vs = np.clip(delta_vs, 1.0, v_impact * 1000)  # can't exceed impact velocity
    
    # Isotropic ejection from collision point
    theta = rng.uniform(0, 2 * np.pi, n_fragments)
    phi = np.arccos(rng.uniform(-1, 1, n_fragments))
    
    dv_x = delta_vs * np.sin(phi) * np.cos(theta) / 1000  # km/s
    dv_y = delta_vs * np.sin(phi) * np.sin(theta) / 1000
    dv_z = delta_vs * np.cos(phi) / 1000

    parent_pos = np.asarray(parent_pos, dtype=float)
    parent_vel = np.asarray(parent_vel, dtype=float)
    
    epoch_days = (jd_epoch + fr_epoch) - 2433281.5
    
    fragments = []
    failed = 0
    
    for i in range(n_fragments):
        frag_vel = parent_vel + np.array([dv_x[i], dv_y[i], dv_z[i]])
        
        # Convert position + velocity to Keplerian elements
        r_vec = parent_pos
        v_vec = frag_vel
        
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        
        # Specific angular momentum
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        
        # Node vector
        k_hat = np.array([0, 0, 1])
        n_vec = np.cross(k_hat, h_vec)
        n = np.linalg.norm(n_vec)
        
        # Eccentricity vector
        e_vec = ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
        ecc = np.linalg.norm(e_vec)
        
        # Skip hyperbolic or near-deorbit fragments
        if ecc >= 1.0:
            failed += 1
            continue
        
        # Semi-major axis
        a = 1 / (2/r - v**2/mu)
        if a <= 0 or a * (1 - ecc) < 6371 + 100:  # perigee below 100 km = reentry
            failed += 1
            continue
        
        # Inclination
        inc = math.acos(np.clip(h_vec[2] / h, -1, 1))
        
        # RAAN
        if n > 1e-10:
            raan = math.acos(np.clip(n_vec[0] / n, -1, 1))
            if n_vec[1] < 0:
                raan = 2 * math.pi - raan
        else:
            raan = 0.0
        
        # Argument of perigee
        if n > 1e-10 and ecc > 1e-10:
            argp = math.acos(np.clip(np.dot(n_vec, e_vec) / (n * ecc), -1, 1))
            if e_vec[2] < 0:
                argp = 2 * math.pi - argp
        else:
            argp = 0.0
        
        # True anomaly
        if ecc > 1e-10:
            ta = math.acos(np.clip(np.dot(e_vec, r_vec) / (ecc * r), -1, 1))
            if np.dot(r_vec, v_vec) < 0:
                ta = 2 * math.pi - ta
        else:
            ta = 0.0
        
        # True anomaly to mean anomaly
        E = 2 * math.atan2(math.sqrt(1 - ecc) * math.sin(ta / 2),
                           math.sqrt(1 + ecc) * math.cos(ta / 2))
        M = E - ecc * math.sin(E)
        if M < 0:
            M += 2 * math.pi
        
        # Mean motion
        n_motion = math.sqrt(mu / a**3) * 60  # rad/min for sgp4
        
        bstar = 2.2 * am_ratios[i] / (2 * 6371.0)  # crude conversion, primarily useful for broad stastistics sought in this work
        
        sat = Satrec()
        sat.sgp4init(
            WGS72, 'i',
            80000 + i,
            epoch_days,
            bstar,
            0.0, 0.0,
            ecc,
            argp,
            inc,
            M,
            n_motion,
            raan,
        )
        
        if sat.error == 0:
            fragments.append(sat)
        else:
            failed += 1
    
    print(f"Created {len(fragments)} fragment Satrec objects ({failed} discarded)")
    return fragments