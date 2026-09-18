# 🤖 RoboMind AI – Uncertainty-Aware Warehouse Robot Brain

> An AI-based warehouse robot system that combines planning, uncertainty reasoning, game playing, connectionist models, and an expert maintenance system for intelligent warehouse operations.

---

# 📖 About The Project

**RoboMind AI** is an implementation-based AI project that simulates the decision-making system of a warehouse robot. The project combines five major AI areas: planning, reasoning under uncertainty, adversarial game playing, connectionist models, and expert systems. It demonstrates how a robot can plan box-stacking tasks, reason about damaged boxes using multiple uncertain-reasoning methods, compete for a charging dock, detect anomalies in sensor data, and diagnose robot faults. The project also includes an integrated control loop and a real industrial anomaly-detection dataset for the optional connectionist-model stretch goal.

---

# ✨ Features

## 🧠 Planning

* Classic Goal Stack Planning for Blocks World
* Arbitrary start and goal states using JSON files
* Nonlinear planning using constraints
* Hierarchical planning with high-level pallet operations
* Reactive planning when a disturbance occurs
* Execution trace and reactive intervention logs

---

## 🔍 Uncertainty Reasoning

* Nonmonotonic reasoning
* Bayes' theorem and posterior probability
* MYCIN-style Certainty Factors
* Bayesian Network inference
* Dempster-Shafer evidence combination
* Fuzzy logic and defuzzification
* Comparison of multiple reasoning methods on synthetic sensor readings

---

## 🎮 Game Playing

* MiniMax algorithm
* Alpha-Beta pruning
* Node-count comparison
* Iterative Deepening Search
* Move ordering heuristic
* Decision-time measurement
* Matplotlib node-exploration benchmark
* Playable dock-contestion game

---

## 🔗 Connectionist Models

* Hopfield Network implemented from scratch
* 8×8 binary patterns
* 10–15% corrupted-bit associative recall
* GRU-based recurrent neural network
* Prediction from the last 10 sensor readings
* Held-out anomaly detection evaluation
* Real SKAB industrial anomaly dataset support

---

## 🛠️ Expert System

* Rule-based robot fault diagnosis
* 18 maintenance rules
* Forward-chaining inference
* Explanation facility and rule trace
* Knowledge acquisition through JSON
* Add new rules without modifying the Python engine

---

# 🛠️ Tech Stack

| Area                 | Technology                              |
| -------------------- | --------------------------------------- |
| **Language**         | Python                                  |
| **Planning**         | Custom Python planning algorithms       |
| **Uncertainty**      | Python, NumPy, Bayesian/Fuzzy reasoning |
| **Bayesian Network** | pgmpy                                   |
| **Fuzzy Logic**      | scikit-fuzzy                            |
| **Game Playing**     | Python, MiniMax, Alpha-Beta             |
| **Visualization**    | Matplotlib                              |
| **Hopfield Network** | NumPy                                   |
| **RNN**              | PyTorch / GRU                           |
| **Expert System**    | Hand-rolled forward-chaining engine     |
| **Dataset**          | SKAB industrial anomaly dataset         |

---

# 📂 Project Structure

```text
RoboMind_AI/
│
├── planning/
│   ├── plan.py
│   ├── planner.py
│   ├── start.json
│   └── goal.json
│
├── uncertainty/
│   ├── reasoning.py
│   └── run_uncertainty.py
│
├── game/
│   ├── dock_game.py
│   ├── benchmark.py
│   └── search_benchmark.ipynb
│
├── connectionist/
│   ├── hopfield.py
│   ├── rnn_anomaly.py
│   ├── connectionist_demo.py
│   └── anomaly_recall.ipynb
│
├── expert/
│   ├── expert_advisor.py
│   └── rules.json
│
├── data/
│   └── skab/
│       └── valve1/
│           ├── 0.csv
│           ├── 1.csv
│           └── ...
│
├── test_cases/
│   ├── run_tests.py
│   ├── start_case1.json
│   ├── goal_case1.json
│   ├── start_case2.json
│   └── goal_case2.json
│
├── main.py
├── nodes_explored.png
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## Prerequisites

* Python 3.10 or newer
* pip
* Internet connection for installing Python dependencies
* PyTorch for the RNN module

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Devansh-Mankad/RoboMind_AI.git
cd RoboMind_AI
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running The Project

## Module A – Planning

Run the planner using the provided start and goal states:

```bash
python -m planning.plan planning/start.json planning/goal.json
```

The planner displays the generated plan, execution trace, and reactive intervention information.

---

## Module B – Uncertainty Reasoning

Run the uncertainty reasoning demonstration:

```bash
python uncertainty/run_uncertainty.py
```

It compares nonmonotonic reasoning, Bayes, Certainty Factors, Bayesian Network inference, Dempster-Shafer reasoning, and fuzzy logic using the sensor dataset.

---

## Module C – Game Playing

Run the playable dock-contestion game:

```bash
python game/dock_game.py --depth 5
```

Run the search benchmark:

```bash
python game/benchmark.py
```

The benchmark compares MiniMax, Alpha-Beta, and Alpha-Beta with Iterative Deepening across different search depths.

---

## Module D – Connectionist Models

Run the connectionist demonstration:

```bash
python connectionist/connectionist_demo.py
```

The Hopfield section demonstrates associative recall from corrupted binary patterns.

For the RNN anomaly detector using the SKAB dataset:

```bash
python connectionist/rnn_anomaly.py --dataset data/skab/valve1
```

The RNN uses sequences of the last 10 sensor readings to predict anomalies.

---

## Module E – Expert System

Run the maintenance advisor:

```bash
python expert/expert_advisor.py
```

The system accepts robot symptoms and provides fault diagnoses together with the rules that fired.

The knowledge base is stored in:

```text
expert/rules.json
```

New rules can be added through the knowledge-acquisition interface without modifying the inference engine.

---

# 🔬 Module Demonstrations

## Module A – Planning

Module A demonstrates AI planning for a Blocks World warehouse pallet using Goal Stack Planning, nonlinear planning, hierarchical planning, and reactive behavior. The planner works with configurable start and goal states and produces an execution trace. A disturbance such as a knocked-over box can trigger the reactive layer during execution instead of restarting the complete planning process.

## Module B – Reasoning Under Uncertainty

Module B demonstrates multiple approaches for deciding whether a warehouse box is damaged when sensor information is uncertain or conflicting. It implements nonmonotonic reasoning, Bayes, Certainty Factors, Bayesian Networks, Dempster-Shafer Theory, and fuzzy logic. The methods are evaluated on the same synthetic sensor readings so their different conclusions can be compared.

## Module C – Game Playing

Module C models charging-dock contention as an adversarial game between robots. It demonstrates plain MiniMax, Alpha-Beta pruning, Iterative Deepening, and move ordering while measuring the number of explored nodes and decision time. A Matplotlib benchmark compares search performance across depths 2–6, and the module also provides a playable command-line game.

## Module D – Connectionist Models

Module D demonstrates two connectionist approaches to warehouse sensor and label data. A Hopfield Network implemented from scratch performs associative recall of corrupted 8×8 binary patterns, while a GRU-based RNN predicts anomalies from the last 10 sensor readings. The RNN can be evaluated on the SKAB industrial anomaly dataset and reports held-out classification metrics.

## Module E – Expert System

Module E implements a rule-based maintenance advisor for diagnosing robot faults from symptom flags. The system contains 18 maintenance rules and uses forward chaining to reach conclusions. It also provides a rule trace explaining why a diagnosis was produced and includes a knowledge-acquisition mechanism for adding rules through the JSON knowledge base.

---

# 🔗 Integrated System

The main control program connects the five AI modules into a simulated warehouse-robot workflow.

```text
                 Synthetic Sensor Events
                         │
                         ▼
              ┌──────────────────────┐
              │ Module B             │
              │ Uncertainty Reasoning│
              └──────────┬───────────┘
                         │
                 Damage suspected?
                         │
                         ▼
              ┌──────────────────────┐
              │ Module E             │
              │ Expert Diagnosis     │
              └──────────────────────┘

       ┌─────────────────────────────────────┐
       │ Module A – Pallet Planning          │
       │ Stacking + Reactive Disturbances    │
       └─────────────────────────────────────┘

       ┌─────────────────────────────────────┐
       │ Module C – Dock Contention          │
       │ MiniMax + Alpha-Beta + IDS          │
       └─────────────────────────────────────┘

       ┌─────────────────────────────────────┐
       │ Module D – Anomaly Detection        │
       │ Hopfield + GRU/RNN sensor analysis  │
       └─────────────────────────────────────┘
```

Run the complete project with:

```bash
python main.py
```

The integrated simulation uses a synthetic warehouse shift and demonstrates the interaction between the different AI modules.

---

# 🧪 Testing

The project includes automated tests covering all five modules.

Run:

```bash
python test_cases/run_tests.py
```

Expected result:

```text
Running 2 test cases per module...
Module A: 2/2 passed
Module B: 2/2 passed
Module C: 2/2 passed
Module D: 2/2 passed
Module E: 2/2 passed
ALL TESTS PASSED (10/10)
```

---

# 📊 RNN Results on SKAB

The optional SKAB dataset implementation was used to evaluate the GRU anomaly detector on multiple industrial sensor experiments.

Example held-out evaluation:

```text
Accuracy  : 0.907
Precision : 0.874
Recall    : 0.867
F1 Score  : 0.871

Confusion Matrix:
TP = 1047
TN = 2001
FP = 151
FN = 160
```

The dataset contains industrial sensor measurements including:

```text
Accelerometer1RMS
Accelerometer2RMS
Current
Pressure
Temperature
Thermocouple
Voltage
Volume Flow RateRMS
```

The dataset files are expected to use semicolon-separated CSV values.

---

# 🎯 Project Requirements Covered

The project covers the five major AI areas specified in the assignment:

* **Module A:** Planning and reactive systems
* **Module B:** Reasoning under uncertainty
* **Module C:** Game playing and adversarial search
* **Module D:** Connectionist models and anomaly detection
* **Module E:** Expert systems and knowledge acquisition
* **Integration:** Combined warehouse robot control workflow
* **Stretch Goal:** Real industrial SKAB dataset for Module D

---

# 👨‍💻 Author

**Devansh Mankad**

Computer Engineering Student

* GitHub: https://github.com/Devansh-Mankad
---

# ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

---

# 📄 License

This project is licensed under the MIT License.