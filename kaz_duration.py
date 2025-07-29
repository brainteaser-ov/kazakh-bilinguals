import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


kaz_path = Path('/your file path xlsx')
rus_path = Path('/your file path xlsx')


def speaker_id(fp: str) -> str:
    stem = Path(fp).stem
    parts = stem.split('_')
    return '_'.join(parts[:-1])


def prep(df: pd.DataFrame, label: str) -> pd.DataFrame:
    vowels = {'a','e','i','o','u'}    # + добавьте нужные символы
    df = df[df['Sound Name'].isin(vowels)].copy()
    df['Duration_ms'] = df['Duration'] * 1000     # сек ➜ мс
    df['Speaker'] = df['File Name'].apply(speaker_id)
    df['Language'] = label
    return df[['Language','Speaker','Sound Name','Duration_ms']]

data = pd.concat([prep(pd.read_excel(kaz_path), 'Казахский(L1)'),
                  prep(pd.read_excel(rus_path), 'Русский (L2)')])


spk_means = (data.groupby(['Language','Speaker','Sound Name'])
                  .Duration_ms.mean()
                  .rename('spk_mean_ms')
                  .reset_index())

agg = (spk_means.groupby(['Language','Sound Name'])
                 .agg(mean_ms=('spk_mean_ms','mean'),
                      sd=('spk_mean_ms','std'),
                      n=('spk_mean_ms','count'))
                 .reset_index())
agg['ci95'] = 1.96 * agg['sd']/np.sqrt(agg['n'])


vowel_order = ['i','e','a','o','u']

fig, ax = plt.subplots(figsize=(10,5))
for lang, sub in agg.groupby('Language'):
    means = [sub.loc[sub['Sound Name']==v,'mean_ms'].values[0]
             if v in sub['Sound Name'].values else np.nan
             for v in vowel_order]
    cis   = [sub.loc[sub['Sound Name']==v,'ci95'].values[0]
             if v in sub['Sound Name'].values else 0
             for v in vowel_order]
    ax.errorbar(range(len(vowel_order)), means, yerr=cis,
                marker='o', linestyle='-', label=lang)

ax.set_xticks(range(len(vowel_order)))
ax.set_xticklabels(vowel_order)
ax.set_ylabel('Средняя длительность гласного (ms)')
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
