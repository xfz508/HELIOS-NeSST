import matplotlib.pyplot as plt

# Data organized by target variable
metrics = [
    {
        "name": "Yn, fD=1.0",
        "helios": 15.8,
        "obs_20_24": (18.7, 9.0),
        "calc_20_24": (44.6, 18.7),
        "scale": "Yn, x10¹⁰",
    },
    {
        "name": "Yn, fD=0.5",
        "helios": 4.22,
        "obs_20_24": (1.29, 0.58),
        "calc_20_24": (6.29, 2.80),
        "scale": "Yn, x10¹⁰",
    },
    {
        "name": "Yp, fD=0.5",
        "helios": 11.0,
        "obs_20_24": (6.28, 1.46),
        "calc_20_24": (18.4, 4.22),
        "scale": "Yp, x10⁸",
    },
]

# Create subplots for each measurement variable
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
i = 0
for ax, m in zip(axes, metrics):
    # 1. Dataset: Observed (20 & 24 µm pair)
    ax.plot(
        [20, 24],
        m["obs_20_24"],
        'x',
        color='c',
        linewidth=2,
        markersize=8,
        label='Rygg (20 & 24 µm)',
    )

    # 2. Dataset: Calculated (20 & 24 µm pair)
    ax.plot(
        [20, 24],
        m["calc_20_24"],
        '+',
        color='b',
        linewidth=2,
        markersize=8,
        label='LILAC (20 & 24 µm)',
    )

    # 3. Dataset: HELIOS (15 µm single dataset point)
    ax.plot(
        15,
        m["helios"],
        'o',
        color='r',
        markersize=10,
        label='HELIOS (15 µm)',
    )

    # Axes styling
    ax.set_title(m["name"], fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Shell Thickness (µm)", fontsize=10)
    print(ax)
    ax.set_ylabel(m["scale"], fontsize=10)
    ax.set_xticks([15, 20, 24])
    ax.set_xlim(13, 26)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=9, loc='upper left')
    i = i+1
"""
plt.suptitle(
    "HELIOS (15 µm) vs 20–24 µm Observed & Calculated Pairs",
    fontsize=14,
    fontweight='bold',
    y=1.03,
)
"""
plt.tight_layout()
plt.show()