"""Generate sample charts (with known ground truth) for demoing the UniChart ChartQA model.

Run once:  python make_examples.py
Writes PNGs + examples.json into ./examples
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "examples"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


examples = []

# 1. Vertical bar chart
countries = ["USA", "China", "India", "Germany", "Brazil"]
values = [62, 48, 35, 27, 19]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(countries, values, color="#3b6fd8")
ax.bar_label(bars)
ax.set_title("Smartphone users who shop online (%)")
ax.set_ylabel("Share of users (%)")
examples.append({
    "file": save(fig, "bar_online_shopping.png"),
    "qa": [
        ["Which country has the highest value?", "USA"],
        ["What is the value for India?", "35"],
        ["What is the difference between USA and Brazil?", "43"],
    ],
    "truth": dict(zip(countries, values)),
})

# 2. Line chart
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
revenue = [12.4, 14.1, 15.8, 11.2, 16.9, 19.3, 21.0]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(years, revenue, marker="o", color="#d8573b", linewidth=2.5)
for x, y in zip(years, revenue):
    ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_title("Annual revenue (billion USD)")
ax.set_xlabel("Year")
ax.set_ylabel("Revenue")
examples.append({
    "file": save(fig, "line_revenue.png"),
    "qa": [
        ["In which year was the revenue the lowest?", "2020"],
        ["What was the revenue in 2022?", "19.3"],
        ["What is the average revenue from 2021 to 2023?", "19.07"],
    ],
    "truth": dict(zip(map(str, years), revenue)),
})

# 3. Pie chart
labels = ["Coal", "Natural gas", "Nuclear", "Renewables", "Other"]
shares = [36, 23, 10, 29, 2]
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.pie(shares, labels=labels, autopct="%d%%", startangle=90,
       colors=["#555555", "#e0a030", "#8a5cd8", "#3bb273", "#bbbbbb"])
ax.set_title("Global electricity generation by source")
examples.append({
    "file": save(fig, "pie_electricity.png"),
    "qa": [
        ["What percentage of electricity comes from renewables?", "29"],
        ["Which source has the largest share?", "Coal"],
        ["Is the share of nuclear greater than natural gas?", "No"],
    ],
    "truth": dict(zip(labels, shares)),
})

# 4. Grouped horizontal bar chart
groups = ["18-29", "30-49", "50-64", "65+"]
men = [71, 64, 52, 38]
women = [76, 69, 55, 41]
fig, ax = plt.subplots(figsize=(7.5, 5))
y = range(len(groups))
b1 = ax.barh([i + 0.2 for i in y], men, height=0.4, label="Men", color="#2a9d8f")
b2 = ax.barh([i - 0.2 for i in y], women, height=0.4, label="Women", color="#e76f51")
ax.bar_label(b1)
ax.bar_label(b2)
ax.set_yticks(list(y), groups)
ax.set_title("Adults who use social media daily, by age (%)")
ax.legend()
examples.append({
    "file": save(fig, "grouped_social_media.png"),
    "qa": [
        ["What is the value for women aged 65+?", "41"],
        ["Which age group has the highest share of men?", "18-29"],
        ["What is the sum of men and women in the 30-49 group?", "133"],
    ],
    "truth": {"men": dict(zip(groups, men)), "women": dict(zip(groups, women))},
})

(OUT / "examples.json").write_text(json.dumps(examples, indent=2))
print(f"Wrote {len(examples)} example charts to {OUT}")
