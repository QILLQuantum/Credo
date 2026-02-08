def guided_trickle_growth(self):
    growth = 0
    H_variance = np.var(list(self.bond_entropy_h.values()) + 
                        list(self.bond_entropy_v.values()) + 
                        list(self.bond_entropy_z.values()))
    
    mischief_factor = 1.0
    if H_variance > 0.05:
        mischief_factor = 1.5  # Padfoot wild mode
    else:
        mischief_factor = 0.8  # Prongs precise mode

    for dir_key, map_dict, entropy_dict in [
        ('h', self.bond_map_h, self.bond_entropy_h),
        ('v', self.bond_map_v, self.bond_entropy_v),
        ('z', self.bond_map_z, self.bond_entropy_z)
    ]:
        for key in map_dict:
            entropy = entropy_dict.get(key, 0.0)
            prob = min(0.95, 0.2 + 0.75 * entropy * mischief_factor)
            if np.random.rand() < prob:
                curr = map_dict[key]
                new = min(curr + 4, 48)
                if new != curr:
                    map_dict[key] = new
                    growth += 1
    return growth