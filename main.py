import random
import numpy as np
from planning.planner import gsp_plan, nonlinear_plan, execute_with_reactive, goal_reached
from uncertainty.reasoning import READINGS, bayes, certainty_factor, dempster_shafer, fuzzy_score, bayesian_network
from game.dock_game import DockGame, Search, chart
from connectionist.hopfield import demo as hopfield_demo
from connectionist.rnn_anomaly import train_model, predict_stream
from expert.expert_advisor import diagnose


def build_shift(events=60, seed=21):
    rng = random.Random(seed)
    readings = []

    for i in range(events):
        readings.append({"id": i + 1, "weight_deviation": rng.random(), "crack_severity": rng.random(), "tilt": int(rng.random() < 0.25)})

    for start in (20, 45):
        for j in range(start, min(start + 3, events)):
            readings[j]["weight_deviation"] = 0.90
            readings[j]["crack_severity"] = 0.85
            readings[j]["tilt"] = 1

    return readings


def run_shift(model, events=60, seed=21):
    readings = build_shift(events, seed)
    damage_hits = 0; anomaly_hits = []

    for r in readings:
        posterior = bayes(r)
        if posterior >= 0.5:
            damage_hits += 1
            symptoms = []

            if r["weight_deviation"] > 0.65: symptoms.append("weight_unstable")
            if r["crack_severity"] > 0.60: symptoms.append("vision_inconsistent")
            if r["tilt"]: symptoms.append("tilt_triggered")

            diagnoses, trace = diagnose(symptoms or ["vision_inconsistent"])
            print(f"Event {r['id']}: damage={posterior:.2f}; maintenance={diagnoses or ['check sensors']}; rules={len(trace)}")

    for end, probability, flag in predict_stream(model, readings):
        if flag:
            anomaly_hits.append((end, probability))

    print(f"Shift events: {events}; suspected damaged boxes: {damage_hits}; RNN anomaly windows: {len(anomaly_hits)}")

    if anomaly_hits:
        print("RNN anomaly examples:", [(e, round(p, 2)) for e, p in anomaly_hits[:5]])
    return readings, anomaly_hits


def main():
    start = {"on": {"B":"A","D":"C","F":"E"}, "ontable":["A","C","E","G"]}
    goal = {"on": {"A":"B","C":"D","E":"F"}, "ontable":["B","D","F","G"]}
    plan = gsp_plan(start, goal); _, layers = nonlinear_plan(start, goal)
    trace, interventions, final_state = execute_with_reactive(start, plan, goal=goal, disturbance_at=max(2, len(plan)//2))

    print("=== Module A ===")
    print("Goal Stack actions:", len(plan), "| Nonlinear layers:", len(layers), "| Sequential actions:", len(plan))
    print("Reactive interventions:", len(interventions), "| Goal reached:", goal_reached(final_state, goal))

    print("\n=== Module B ===")
    for idx in (2, 6):
        r = READINGS[idx]; cf, serial = certainty_factor(r); belief, plausibility = dempster_shafer(r)
        print(f"Case {r['id']}: Bayes={bayes(r):.3f} CF={cf:.3f} BN={bayesian_network(r):.3f} DS=[{belief:.3f},{plausibility:.3f}] Fuzzy={fuzzy_score(r):.3f}")

    print("\n=== Module C ===")
    game = DockGame(); s = Search(game); value, move = s.alphabeta(game.piles, 5)
    print("Alpha-Beta move:", move[:2], "nodes:", s.nodes, "value:", value)
    print("Benchmark:", chart())

    print("\n=== Module D ===")
    for idx in (0, 2):
        _, _, _, hacc, steps = hopfield_demo(index=idx)
        print(f"Hopfield pattern {idx+1}: recall={hacc:.2%}, updates={steps}")
    try:
        rnn_model, rnn_info = train_model()
        rnn_metrics = rnn_info["metrics"]
        print("RNN test metrics:", {k: round(v, 3) for k, v in rnn_metrics.items()})

    except FileNotFoundError as exc:
        rnn_model, rnn_info = None, None
        print("RNN dataset not found:", exc)

    print("\n=== Module E ===")
    print("Battery diagnosis:", diagnose(["battery_voltage_low","charge_cycle_failed","battery_temperature_high"])[0])

    print("\n=== Integrated shift ===")
    if rnn_model is not None:
        print("Module D uses the manually downloaded SKAB dataset; run the Module D script with --dataset to evaluate another CSV.")

if __name__ == "__main__": main()