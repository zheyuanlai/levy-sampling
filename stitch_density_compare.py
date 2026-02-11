#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "density_compare")

ORDER = [
    ["fourwell_true_density.png", "fourwell_diffusion_density.png", "fourwell_mala_density.png", "fourwell_levy_density.png"],
    ["mueller_true_density.png", "mueller_diffusion_density.png", "mueller_mala_density.png", "mueller_levy_density.png"],
    ["ring_true_density.png", "ring_diffusion_density.png", "ring_mala_density.png", "ring_levy_density.png"],
]

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing image: {path}")
    return mpimg.imread(path)

def main():
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(3):
        for j in range(4):
            img_path = os.path.join(SRC_DIR, ORDER[i][j])
            img = load_image(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    out_path = os.path.join(SRC_DIR, "density_compare_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
