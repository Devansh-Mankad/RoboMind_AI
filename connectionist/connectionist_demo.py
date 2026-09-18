from hopfield import demo
from rnn_anomaly import train_model

def main():
    original, noisy, recalled, accuracy, steps = demo()
    print("Hopfield recall accuracy:", f"{accuracy:.2%}", "steps:", steps)
    try:
        _, info = train_model()
        print("RNN test metrics:", {k: round(v, 3) for k, v in info["metrics"].items()})
    except FileNotFoundError as exc:
        print("RNN dataset not found:", exc)


if __name__ == "__main__":
    main()