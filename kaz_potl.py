
from pathlib import Path

file_path = Path('/your file path')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Ellipse


df = pd.read_excel(file_path)

sound_col = df.columns[3]
f1_col = df.columns[5]
f2_col = df.columns[6]


vowels = df[~df[sound_col].str.strip().str.lower().isin(['t', "t'", 'f', 'j'])].copy()
vowels['F1'] = pd.to_numeric(vowels[f1_col], errors='coerce')
vowels['F2'] = pd.to_numeric(vowels[f2_col], errors='coerce')
vowels['Sound'] = vowels[sound_col].str.strip()
vowels = vowels.dropna(subset=['F1', 'F2'])

sounds = vowels['Sound'].unique()
palette = dict(zip(sounds, sns.color_palette('tab10', n_colors=len(sounds))))

plt.figure(figsize=(8, 8))
ax = plt.gca()

for sound in sounds:
    sub = vowels[vowels['Sound'] == sound]
    color = palette[sound]

    ax.scatter(sub['F2'], sub['F1'], s=10, color=color, alpha=0.6)

    cov = np.cov(sub['F2'], sub['F1'])
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellipse = Ellipse(xy=(sub['F2'].mean(), sub['F1'].mean()),
                      width=width, height=height, angle=theta,
                      edgecolor=color, facecolor='none', linewidth=1)
    ax.add_patch(ellipse)

    ax.text(sub['F2'].mean(), sub['F1'].mean(), sound,
            color=color, fontsize=12, ha='center', va='center',
            weight='bold')


ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel('F2, Гц')
ax.set_ylabel('F1, Гц')
plt.tight_layout()
plt.show()
