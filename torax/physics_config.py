# Physics data generation config
CONFIG = {
    'profile_conditions': {
        'Ip': 15.0e6,  # 15 MA plasma current
        'T_i': {0.0: {0.0: 10.0, 0.5: 5.0, 1.0: 0.5}},  # Ion temperature profile (keV)
        'T_i_right_bc': 0.5,
        'T_e': {0.0: {0.0: 12.0, 0.5: 6.0, 1.0: 0.8}},  # Electron temperature profile (keV)
        'T_e_right_bc': 0.8,
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': 0.3,
        'n_e_nbar_is_fGW': True,
        'nbar': 0.85,  # Line-averaged density
        'n_e': {0: {0.0: 1.0, 0.5: 0.8, 1.0: 0.3}},  # Electron density profile
    },
    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'Z_eff': 2.0,
    },
    'numerics': {
        't_final': 3.0,  # 3 second simulation
        'fixed_dt': 0.05,  # Fine time resolution
    },
    'geometry': {
        'geometry_type': 'circular',
        'R_major': 6.2,
        'a_minor': 2.0,
        'B_0': 5.3,
        'elongation_LCFS': 1.8,  # Slightly elongated
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        'generic_current': {},
        'generic_particle': {
            'S_puff_val': 2.0e20,  # Particle source
            'puff_decay_length': 0.05,
        },
        'gas_puff': {},
        'pellet': {},
        'generic_heat': {
            'gaussian_location': 0.25,
            'gaussian_width': 0.15,
            'P_total': 75.0e6,  # 75 MW heating power
            'electron_heat_fraction': 0.67,
        },
        'fusion': {},
        'ei_exchange': {},
        'ohmic': {},
    },
    'pedestal': {},
    'transport': {
        'model_name': 'constant',
        # Add transport coefficients
        'chi_e': 1.0,  # Electron heat diffusivity
        'chi_i': 1.0,  # Ion heat diffusivity
        'D_e': 0.5,    # Particle diffusivity
    },
    'solver': {
        'solver_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
    # Output configuration
    'diagnostics': {
        'writedata': True,
        'dt_data': 0.1,  # Output every 0.1 seconds
    },
}