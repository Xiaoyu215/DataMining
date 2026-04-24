# 🧐 Beyond Weird: Context-Aware Detection of Suspicious Music Streaming

> Streaming-music fraud costs artists and labels an estimated **$2B a year** — but real listening is so heterogeneous that simple "weird = bot" rules mostly catch humans having a weird week. This project builds a **context-aware triage system** that tells *scam* apart from *strange* on 50M events of real listening data.

👉 **The main deliverable is [`Main.ipynb`](https://github.com/Xiaoyu215/DataMining/blob/main/Main.ipynb)** — start there.

🎬 **Introduction video:** [▶ Watch the intro](https://youtu.be/SYTr_wjy1Lo)

🌎 **Website:** [Quick look here]()



---

## 🧪 Research Questions

This project answers three layered questions, each building on the last:

1. **🏘️ RQ1 — Who are the listeners?** Can we segment users into interpretable behavioral archetypes that are stable enough to anchor downstream anomaly detection?
2. **🤖 RQ2 — What does "anomalous" mean inside each archetype?** Does per-cluster anomaly detection surface different, more meaningful patterns than a single global model?
3. **🌊 RQ3 — Is suspiciousness persistent or state-dependent?** When a user's listening mode shifts from week to week, do their anomaly flags follow them, or do they dissolve?

The layering is the point: **you can only detect what deviates from normal once you've mapped what normal looks like — and accepted that "normal" is plural.**

---

## 💡 Results at a glance

- **Three stable listener archetypes** — Lean-back loyal (53%), Casual explorers (27%), Active samplers (19%) — survive four clustering algorithms, reseeding, a first-half / second-half time split, and feature-removal stress tests.
- **Anomaly is archetype-specific.** A loyal listener gets flagged for being *too loyal* (one track absorbing 90%+ of plays); an explorer for being *too optimized* (metronome timing, wide-taste mask over a single repeated item); a sampler for sampling the *same thing on a loop*. A single global Isolation Forest misses most of these.
- **81% of weekly anomaly flags dissolve on their own** — 64% are transient (weird week), 17% are state-dependent (listening mode shifted). A naive "flag and ban" policy would mostly punish humans.
- **~80 users out of ~5,600 remain suspicious across time and state transitions.** A manageable review queue, not a mass-labeling problem.
- **The final output is a stratified 5-tier framework**, not a single score — T1 top priority (tiny, high-confidence), down through T5 clean (the vast majority).

---

## 📂 Repo Structure

```
BeyondWeird/
├── Main.ipynb              ← 👉 main deliverable (the full story)
├── preprocessing.py        ← all data-prep code (RQ1 + RQ2 + RQ3), importable
├── requirements.txt        ← pinned dependencies
├── README.md               ← you are here
├── data/                   ← parquet cache (auto-created on first run)
│   ├── listens.parquet
│   ├── sess.parquet
│   └── user_features.parquet
└── checkpoints/            ← intermediate deliverables from earlier in the project
    ├── checkpoint1_eda.ipynb       ← dataset comparison + initial Yambda EDA
    └── checkpoint2_rqs.ipynb       ← research-question definition
```

---

## 📊 Data

**Dataset:** [Yandex Yambda](https://huggingface.co/datasets/yandex/yambda), 50M-event multi-event flat parquet.

**Source URL:** `hf://datasets/yandex/yambda/flat/50m/multi_event.parquet` (pulled directly via DuckDB, no manual download needed)

**What the data contains:** per-event records of user interactions with the Yandex Music catalog — listens, likes, dislikes, unlikes, undislikes — with timestamps, item IDs, organic/algorithmic entry flags, and played-ratio.

**Preprocessing lives in `preprocessing.py`** and is fully automated. On first run, the pipeline:
1. Streams the raw parquet through DuckDB
2. Sessionizes listen events (30-minute idle → new session)
3. Computes per-user, per-session, and entropy features
4. Adds anomaly-specific signals (inter-event gap regularity, item concentration, event-mix)
5. Builds weekly (user, week) panels for RQ3
6. Caches everything to `./data/` as parquet

**First run is slow (~few minutes); subsequent runs load from cache in seconds.**

---

## 🚀 How to Reproduce

This project was built and tested in **Google Colab**. To reproduce:

### 1. Get the files into Google Drive
Upload `Main.ipynb`, `preprocessing.py`, and `requirements.txt` into a folder on Google Drive (e.g. `MyDrive/BeyondWeird/`).

### 2. Open in Colab
Right-click `Main.ipynb` in Drive → Open with → Google Colaboratory.

### 3. Update the bootstrap cell
The first code cell mounts Google Drive and `cd`s into the project folder. Update `PROJECT_DIR` to match your folder path.

### 4. Run All
The second code cell installs dependencies. The third imports them. After that, everything runs top-to-bottom. The `./data/` cache persists in Drive across Colab sessions, so you only pay the multi-minute download cost *once*.

> **Run order:** just run `Main.ipynb` top to bottom. There's nothing else to execute.

### Running locally (alternative)
```bash
pip install -r requirements.txt
jupyter notebook Main.ipynb
```

---

## 📦 Key Dependencies

| Package | Version | Why |
|---|---|---|
| `python` | 3.12.13 | Base runtime |

Full pinned list in [`requirements.txt`](./requirements.txt) 🌱

---

## 🪜 Project Timeline

| Checkpoint | What | Output |
|---|---|---|
| 🫧 **Checkpoint 1** | Dataset comparison (FoodPuzzle / Yambda / ViClaim) on tasks, quality, feasibility, bias, ethics, plus initial Yambda EDA: missingness, sparsity, organic vs non-organic, sessionization, transition probabilities, repeat listening | Chose Yambda · initial insights · direction for RQs |
| 🐰 **Checkpoint 2** | Define research questions that combine course techniques (K-Means, Isolation Forest) with externally-learned techniques (per-cluster anomaly, weekly panel design, persistence typology) | Three layered RQs (above) |
| 🎤 **Final** | Execute the full three-RQ analysis and ship a stratified risk framework | `Main.ipynb` |

---

## ✨ TL;DR

> **The best way to find fake listening is to first understand real listening.**
>
> We didn't try to build a detector. We built a **triage system**: five tiers ranking users by how much converging evidence stacks up against them. T1 is the tiny, high-confidence queue worth human review. Everything below T2 is essentially "don't touch" — which is a feature, not a limitation. In fraud detection, being conservative is being correct. 🎯
