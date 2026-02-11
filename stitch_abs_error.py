#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "abs_error")

ORDER = [
    ["fourwell_abs_err_diffusion.png", "mueller_abs_err_diffusion.png", "ring_abs_err_diffusion.png"],
    ["fourwell_abs_err_levy.png", "mueller_abs_err_levy.png", "ring_abs_err_levy.png"],
]

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing image: {path}")
    return mpimg.imread(path)

def main():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(2):
        for j in range(3):
            img_path = os.path.join(SRC_DIR, ORDER[i][j])
            img = load_image(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    out_path = os.path.join(SRC_DIR, "abs_error_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
