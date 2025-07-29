from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

# стиль и размер
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 13


# ---------- загрузка данных ----------
def load_vowel_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # автоматическое переименование
    rename = {}
    for col in df.columns:
        key = str(col).lower()
        if "sound" in key and "name" in key:
            rename[col] = "vowel"
        elif key.strip() == "f1" or "unnamed: 5" in key:
            rename[col] = "F1"
        elif key.strip() == "f2" or "unnamed: 6" in key:
            rename[col] = "F2"
    df = df.rename(columns=rename)

    for need in ("vowel", "F1", "F2"):
        if need not in df.columns:
            raise ValueError(f"Колонка «{need}» не найдена.")

    df["F1"] = pd.to_numeric(df["F1"], errors="coerce")
    df["F2"] = pd.to_numeric(df["F2"], errors="coerce")
    df = df.dropna(subset=["F1", "F2"])

    df["vowel"] = df["vowel"].astype(str).str.strip().str.lower()
    df = df[df["vowel"].isin(["i", "e", "a", "o", "u"])]

    return df[["vowel", "F1", "F2"]]


# ---------- средние значения ----------
def average_formants(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("vowel")[["F1", "F2"]].mean()


# ---------- площадь многоугольника ----------
def polygon_area(points: List[Tuple[float, float]]) -> float:
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


# ---------- построение ----------
def plot_vowel_space(
    data: Dict[str, pd.DataFrame],
    order: List[str],
) -> Dict[str, float]:
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(range(len(data)))

    for (label, means), color in zip(data.items(), colors):
        present = [v for v in order if v in means.index]
        xs = means.loc[present, "F2"].tolist()
        ys = means.loc[present, "F1"].tolist()

        # линия
        ax.plot(xs + xs[:1], ys + ys[:1], color=color, linewidth=1.5, label=label, zorder=2)
        # точки
        ax.scatter(xs, ys, color=color, s=120, zorder=3, edgecolor="white", linewidth=1.5)

        # подписи: чёрные, жирные, с белым фоном
        for v in present:
            ax.text(
                means.loc[v, "F2"],
                means.loc[v, "F1"] + 20,
                v.upper(),
                fontsize=16,
                ha="center",
                va="center",
                fontweight="bold",
                color="black",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8)
            )

    # оси
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")

    ax.set_xlim(2500, 900)
    ax.set_ylim(300, 800)

    ax.invert_yaxis()
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    fig.tight_layout()
    plt.show()

    # расчёт площадей
    areas = {}
    for label, means in data.items():
        present = [v for v in order if v in means.index]
        points = list(zip(means.loc[present, "F2"], means.loc[present, "F1"]))
        areas[label] = polygon_area(points)
    return areas


def build_vowel_plots(path1: str, path2: str) -> None:
    df1 = load_vowel_data(path1)
    df2 = load_vowel_data(path2)

    means1 = average_formants(df1)
    means2 = average_formants(df2)

    data = {
        Path(path1).stem.replace("vowels_results_", ""): means1,
        Path(path2).stem.replace("vowels_results_", ""): means2,
    }

    tri_areas = plot_vowel_space(data, order=["i", "a", "u"])
    for name, area in tri_areas.items():
        print(f"Площадь треугольника /i-a-u/ ({name}): {area:.0f} Гц²")

    penta_areas = plot_vowel_space(data, order=["i", "e", "a", "o", "u"])
    for name, area in penta_areas.items():
        print(f"Площадь пятиугольника /i-e-a-o-u/ ({name}): {area:.0f} Гц²")


if __name__ == "__main__":
    build_vowel_plots(
        "/your file path",
        "/your file path.xlsx"
    )
