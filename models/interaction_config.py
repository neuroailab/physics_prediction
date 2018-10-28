def physics_network(n_classes, ctrl_dist, nonlin='relu'):
    network_cfg = {
            'phiC': {1: {'dropout': 1.0, 'num_features': 400},
            2: {'dropout': 1.0, 'num_features': 400},
            3: {'dropout': 1.0, 'num_features': 400},
            4: {'dropout': 1.0, 'num_features': 400},
            5: {'dropout': 1.0, 'num_features': 400},
            6: {'dropout': 1.0, 'num_features': 400},
            7: {'activation': 'identity', 'num_features': 100},
            'share_vars': False},
            'phiC_depth': 7,
            'phiF': {1: {'dropout': 1.0, 'num_features': 400},
            2: {'dropout': 1.0, 'num_features': 400},
            3: {'activation': 'identity', 'num_features': 100},
            'share_vars': False},
            'phiF_depth': 3,
            'phiO': {1: {'dropout': 1.0, 'num_features': 400},
            2: {'activation': 'identity', 'num_features': 3},
            'share_vars': False},
            'phiO_depth': 2,
            'phiR': {1: {'dropout': 1.0, 'num_features': 400},
            2: {'dropout': 1.0, 'num_features': 400},
            3: {'dropout': 1.0, 'num_features': 400},
            4: {'dropout': 1.0, 'num_features': 400},
            5: {'dropout': 1.0, 'num_features': 400},
            6: {'activation': 'identity', 'num_features': 100},
            'share_vars': False},
            'phiR_depth': 6,
            'phiS': {1: {'dropout': 1.0, 'num_features': 400},
            2: {'dropout': 1.0, 'num_features': 400},
            3: {'dropout': 1.0, 'num_features': 400},
            4: {'dropout': 1.0, 'num_features': 400},
            5: {'dropout': 1.0, 'num_features': 400},
            6: {'dropout': 1.0, 'num_features': 400},
            7: {'activation': 'identity', 'num_features': 100},
            'share_vars': False},
            'phiS_depth': 7}

    for k0,v0 in network_cfg.items():
        if not isinstance(v0, dict):
            continue
        for k1,v1 in v0.items():
            if not isinstance(v1, dict):
                continue
            if 'activation' not in v1:
                network_cfg[k0][k1]['activation'] = nonlin

    return network_cfg
