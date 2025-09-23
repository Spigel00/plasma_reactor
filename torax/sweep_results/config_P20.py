# Auto-generated config for parameter sweep
# Ip = 15.0 MA, P = 20 MW

CONFIG = {
    'profile_conditions': {
        'Ip': 15000000.0,  # 15.0 MA in Amperes
        'T_i': {0.0: {0.0: 8.0, 1.0: 0.2}},
        'T_i_right_bc': 0.2,
        'T_e': {0.0: {0.0: 8.0, 1.0: 0.2}},
        'T_e_right_bc': 0.2,
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': 0.3,
        'n_e_nbar_is_fGW': True,
        'nbar': 1.0,
        'n_e': {0: {0.0: 1.2, 1.0: 1.0}},
    },
    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'Z_eff': 1.8,
    },
    'numerics': {
        't_final': 5.0,
        'fixed_dt': 0.05,
    },
    'geometry': {
        'geometry_type': 'circular',
        'R_major': 6.2,
        'a_minor': 2.0,
        'B_0': 5.3,
        'elongation_LCFS': 1.72,
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        'generic_current': {},
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        'generic_heat': {
            'gaussian_location': 0.2,
            'gaussian_width': 0.1,
            'P_total': 20000000.0,  # 20 MW in Watts
            'electron_heat_fraction': 0.6,
        },
        'fusion': {},
        'ei_exchange': {},
        'ohmic': {},
    },
    'pedestal': {},
    'transport': {
        'model_name': 'constant',
    },
    'solver': {
        'solver_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
