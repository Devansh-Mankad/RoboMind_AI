"""Two independent test cases for each assignment module (10 checks total)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from planning.planner import gsp_plan, nonlinear_plan, replay_plan, goal_reached, execute_with_reactive
from uncertainty.reasoning import READINGS, bayes, certainty_factor, bayesian_network, dempster_shafer, fuzzy_score
from game.dock_game import DockGame, Search
from connectionist.hopfield import demo as hopfield_demo
from connectionist.rnn_anomaly import train_model, predict_stream
from expert.expert_advisor import diagnose
from main import build_shift


def planning_case(start, goal):
    plan = gsp_plan(start, goal)
    assert goal_reached(replay_plan(start, plan), goal)
    _, layers = nonlinear_plan(start, goal)
    assert len(layers) <= len(plan)
    _, _, final = execute_with_reactive(start, plan, goal=goal, disturbance_at=max(2, len(plan)//2))
    assert goal_reached(final, goal)


def main():
    print("Running 2 test cases per module...")
    planning_case({"on":{"B":"A","D":"C","F":"E"},"ontable":["A","C","E","G"]},{"on":{"A":"B","C":"D","E":"F"},"ontable":["B","D","F","G"]})
    planning_case({"on":{"C":"B","E":"D"},"ontable":["A","B","D","F","G"]},{"on":{"B":"A","D":"C","F":"E"},"ontable":["A","C","E","G"]})
    print("Module A: 2/2 passed")

    for idx in (0, 6):
        r=READINGS[idx]
        assert 0 <= bayes(r) <= 1
        cf,_=certainty_factor(r); assert 0 <= cf <= 1
        assert 0 <= bayesian_network(r) <= 1
        b,p=dempster_shafer(r); assert 0 <= b <= p <= 1
        assert 0 <= fuzzy_score(r) <= 1
    print("Module B: 2/2 passed")

    game=DockGame()
    for depth in (2, 4):
        mm=Search(game); mm.minimax(game.piles,depth)
        ab=Search(game); ab.alphabeta(game.piles,depth)
        assert ab.nodes < mm.nodes
    print("Module C: 2/2 passed")

    for idx in (0, 2):
        *_, accuracy, _ = hopfield_demo(index=idx)
        assert accuracy == 1.0
    model,_=train_model(epochs=3)
    rnn_stream = [
    {
        "Accelerometer1RMS": 0.1,
        "Accelerometer2RMS": 0.1,
        "Current": 1.0,
        "Pressure": 1.0,
        "Temperature": 25.0,
        "Thermocouple": 25.0,
        "Voltage": 220.0,
        "Volume Flow RateRMS": 1.0,
    }
    for _ in range(20)
]

    assert len(predict_stream(model, rnn_stream)) == 11
    print("Module D: 2/2 passed")

    d1,_=diagnose(["wheel_slip","motor_current_high"]); d2,_=diagnose(["battery_voltage_low","charge_cycle_failed","battery_temperature_high"])
    assert d1 and d2
    print("Module E: 2/2 passed")
    print("ALL TESTS PASSED (10/10)")

if __name__ == "__main__": main()
