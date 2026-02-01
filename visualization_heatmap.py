import numpy as np
import matplotlib.pyplot as plt

def plot_bond_dim_cone(bond_grid, title="Bond Dimension Cone (10×10)", filename="bond_dim_cone_heatmap.png"):
    """Plot bond_dim heatmap with cone visualization"""
    plt.figure(figsize=(10, 8))
    im = plt.imshow(bond_grid, cmap='plasma', interpolation='nearest')
    plt.colorbar(im, label='Bond Dimension')
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig(filename)
    plt.close()
    print(f"Bond_dim cone heatmap saved: {filename}")