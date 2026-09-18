import numpy as np

PATTERNS = [
    ["11111110","01001110","11010100","10011110","11111001","01010100","10001101","10000111"],
    ["11110011","00010000","11011010","00101011","01001100","01010110","10000100","10011100"],
    ["00011000","11010010","01010001","00110111","10011010","10111111","00001010","10001110"],
    ["11110111","10101100","11011001","10100000","11010111","10101101","10011100","01111111"],
    ["10001101","11100111","10010011","10111001","11010101","10011001","01000100","01011011"],
    ["01011100","00101010","01011101","11010011","10010100","11001100","01000010","00011110"],
]


def to_vector(rows):
    return np.array([1 if c == "1" else -1 for row in rows for c in row], dtype=float)


def to_bitmap(v):
    return ["".join("1" if x > 0 else "0" for x in v[i:i+8]) for i in range(0, 64, 8)]


class Hopfield:
    """From-scratch Hebbian Hopfield associative memory."""
    def __init__(self):
        self.weights = np.zeros((64, 64))

    def train(self, patterns):
        self.weights = sum(np.outer(p, p) for p in patterns)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def recall(self, state, steps=20):
        state = state.copy()
        history = [state.copy()]
        for _ in range(steps):
            old = state.copy()
            for i in range(len(state)):
                field = np.dot(self.weights[i], state)
                state[i] = 1 if field >= 0 else -1
            history.append(state.copy())
            if np.array_equal(state, old):
                break
        return state, history


def demo(index=2, seed=4, flips=8):
    if not 0 <= index < len(PATTERNS):
        raise ValueError("Pattern index must be 0..5")

    if not 6 <= flips <= 10:
        raise ValueError("Corruption must be 10-15% (6-10 of 64 bits)")

    rng = np.random.default_rng(seed)
    patterns = [to_vector(x) for x in PATTERNS]
    net = Hopfield(); net.train(patterns)
    original = patterns[index].copy()
    noisy = original.copy()
    noisy[rng.choice(64, flips, replace=False)] *= -1
    recalled, history = net.recall(noisy)
    accuracy = float(np.mean(recalled == original))

    return to_bitmap(original), to_bitmap(noisy), to_bitmap(recalled), accuracy, len(history) - 1

if __name__ == "__main__":
    for index in (0, 2):
        original, noisy, recalled, accuracy, steps = demo(index=index)
        print(f"Test pattern {index + 1}: corruption=12.5%, recall={accuracy:.2%}, updates={steps}")