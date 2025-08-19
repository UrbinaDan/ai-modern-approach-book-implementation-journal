# ai-modern-approach-book-implementation-journal

# How to use?
When first creating a environment for each chapter do: 
 ```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy matplotlib pandas
 ```

 
*A learning journal and codebase for **Artificial Intelligence: A Modern Approach** (Russell & Norvig).*

This repository documents my journey through AIMA chapter by chapter.  
It combines:

- 📖 **Journal entries** → notes, reflections, and experiment results  
- 💻 **Clean implementations** → reusable Python modules  
- 🧪 **Projects & experiments** → hands-on applications of concepts  
- ✅ **Tests** → correctness checks and reproducibility  

---

## 📂 Repository Structure

 ```bash
aima-journal/
├── chapters/ # Journal-style notes & per-chapter projects
│ ├── ch01_intro/
│ │ ├── notes.md
│ │ └── projects/
│ ├── ch02_agents/
│ │ ├── notes.md
│ │ ├── simple_reflex.py
│ │ ├── model_based.py
│ │ └── goal_based.py
│ └── ...
│
├── core/ # Clean, reusable implementations
│ ├── agents.py
│ ├── search.py
│ ├── utils.py
│ └── ...
│
├── experiments/ # Larger experiments beyond the book
│ ├── exploration.py
│ ├── dirt_drift.py
│ └── ...
│
├── tests/ # Unit tests
├── README.md
└── requirements.txt

 ```
# 📖 Goals
- Reproduce key algorithms from AIMA
- Keep a personal learning journal of insights
- Provide clean, reusable code for others to study or extend
- Show results via small experiments & tests

# 🤝 Contributing
This repo is primarily a personal learning journey, but issues and pull requests are welcome if you’d like to improve implementations or add experiments.

