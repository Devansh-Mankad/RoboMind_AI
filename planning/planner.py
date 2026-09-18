import json
import random
import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    name: str
    box: str
    target: str = ""
    def __str__(self):
        return f"{self.name}({self.box}{',' + self.target if self.target else ''})"


def normalize(state):
    return {
        "on": dict(state.get("on", {})),
        "ontable": set(state.get("ontable", [])),
        "holding": state.get("holding"),
    }


def state_key(s):
    return (tuple(sorted(s["on"].items())), tuple(sorted(s["ontable"])), s["holding"])


def boxes_in(start, goal):
    return sorted(
        set(start["ontable"]) | set(start["on"]) | set(start["on"].values()) |
        set(goal["ontable"]) | set(goal["on"]) | set(goal["on"].values())
    )


def clear_boxes(s, boxes):
    below = set(s["on"].values())
    return [b for b in boxes if b not in below and b != s["holding"]]


def clear(s, box, boxes):
    return box in clear_boxes(s, boxes)


def holding_box(s):
    return s["holding"]


def legal_actions(s, boxes):
    actions = []
    if s["holding"] is None:
        for x in boxes:
            if x in s["ontable"] and clear(s, x, boxes):
                actions.append(Action("PICKUP", x))
        for x, y in sorted(s["on"].items()):
            if clear(s, x, boxes):
                actions.append(Action("UNSTACK", x, y))
    else:
        x = s["holding"]
        actions.append(Action("PUTDOWN", x))
        for y in clear_boxes(s, boxes):
            if y != x:
                actions.append(Action("STACK", x, y))
    return actions


def apply(s, action):
    ns = normalize(s)
    x = action.box
    if action.name == "PICKUP":
        ns["ontable"].remove(x)
        ns["holding"] = x
    elif action.name == "UNSTACK":
        ns["on"].pop(x)
        ns["holding"] = x
    elif action.name == "PUTDOWN":
        ns["ontable"].add(x)
        ns["holding"] = None
    elif action.name == "STACK":
        ns["on"][x] = action.target
        ns["holding"] = None
    else:
        raise ValueError(f"Unknown action: {action.name}")
    return ns


def predicate_holds(state, predicate, boxes):
    kind, *args = predicate
    if kind == "on": return state["on"].get(args[0]) == args[1]
    if kind == "ontable": return args[0] in state["ontable"]
    if kind == "clear": return clear(state, args[0], boxes)
    if kind == "holding": return state["holding"] == args[0]
    if kind == "handempty": return state["holding"] is None
    return False


def goal_predicates(goal):
    predicates = []
    for x, y in sorted(goal["on"].items()):
        predicates.append(("on", x, y))
    for x in sorted(goal["ontable"]):
        predicates.append(("ontable", x))
    return predicates


def achiever(predicate):
    kind = predicate[0]
    if kind == "on":
        return Action("STACK", predicate[1], predicate[2])
    if kind == "ontable":
        return Action("PUTDOWN", predicate[1])
    if kind == "clear":
        return None
    if kind == "holding":
        return None
    if kind == "handempty":
        return None
    return None


def action_preconditions(action):
    if action.name == "STACK":
        return [("holding", action.box), ("clear", action.target)]
    if action.name == "PUTDOWN":
        return [("holding", action.box)]
    if action.name == "PICKUP":
        return [("ontable", action.box), ("clear", action.box), ("handempty",)]
    if action.name == "UNSTACK":
        return [("on", action.box, action.target), ("clear", action.box), ("handempty",)]
    return []


def clear_achiever(state, box, boxes):
    top = next((x for x, y in state["on"].items() if y == box), None)
    if top is None:
        return None
    return Action("UNSTACK", top, box)


def holding_achiever(state, box, boxes):
    if box in state["ontable"]:
        return Action("PICKUP", box)
    support = state["on"].get(box)
    if support is not None:
        return Action("UNSTACK", box, support)
    return None


def goal_stack_plan(start, goal):
    """Classic goal-stack planner using literals, operators, preconditions and effects."""
    start, goal = normalize(start), normalize(goal)
    boxes = boxes_in(start, goal)
    goals = goal_predicates(goal)
    stack = [("GOAL", g) for g in reversed(goals)]
    plan = []
    seen = set()
    max_iterations = 10000

    for _ in range(max_iterations):
        if not stack:
            return plan
        state = replay_plan(start, plan)
        item = stack[-1]
        signature = (state_key(state), tuple(stack[-6:]))

        if signature in seen:
            raise ValueError("Goal Stack Planner could not find a plan for the supplied states")
        seen.add(signature)

        kind, value = item

        if kind == "GOAL":
            if predicate_holds(state, value, boxes):
                stack.pop()
                continue
            action = achiever(value)

            if action is None:
                if value[0] == "clear":
                    action = clear_achiever(state, value[1], boxes)
                elif value[0] == "holding":
                    action = holding_achiever(state, value[1], boxes)
                elif value[0] == "handempty" and state["holding"] is not None:
                    action = Action("PUTDOWN", state["holding"])

            if action is None:
                raise ValueError(f"No operator can achieve goal {value}")

            stack.pop()
            stack.append(("ACTION", action))

            for precondition in reversed(action_preconditions(action)):
                if not predicate_holds(state, precondition, boxes):
                    stack.append(("GOAL", precondition))

        else:
            action = value
            if not all(predicate_holds(state, p, boxes) for p in action_preconditions(action)):
                stack.pop()
                stack.append(("ACTION", action))
                for precondition in reversed(action_preconditions(action)):
                    if not predicate_holds(state, precondition, boxes):
                        stack.append(("GOAL", precondition))
            else:
                stack.pop()
                plan.append(action)

    raise ValueError("Goal Stack Planner exceeded iteration limit")

def replay_plan(start, plan):
    state = normalize(start)
    for action in plan:
        state = apply(state, action)
    return state

def gsp_plan(start, goal):
    return goal_stack_plan(start, goal)

def nonlinear_plan(start, goal):
    """Partial-order/least-commitment view: derive dependencies only where actions interact."""
    sequential = gsp_plan(start, goal)
    n = len(sequential)
    constraints = {i: set() for i in range(n)}

    for i, first in enumerate(sequential):
        for j in range(i + 1, n):
            second = sequential[j]
            first_items = {first.box, first.target}
            second_items = {second.box, second.target}
            if first_items & second_items or first.name in {"PICKUP", "UNSTACK"} and second.name in {"PICKUP", "UNSTACK"}:
                constraints[j].add(i)

    layers, placed = [], set()
    while len(placed) < n:
        layer = [i for i in range(n) if i not in placed and constraints[i].issubset(placed)]
        if not layer:
            raise ValueError("Cyclic partial-order constraints")
        layers.append([sequential[i] for i in layer])
        placed.update(layer)
    return sequential, layers


def hierarchical_load_pallet(start, goal):
    return {"operator": "LoadPallet(P)", "expands_to": gsp_plan(start, goal)}


def execute_with_reactive(start, plan, goal=None, disturbance_at=None, seed=7):
    """Execute the current plan and locally repair a disturbance without full replanning."""
    random.seed(seed)
    state = normalize(start); goal = normalize(goal or {"on": {}, "ontable": []})
    trace, interventions = [], []
    plan = list(plan); index = 0; step = 0

    while index < len(plan):
        step += 1; action = plan[index]
        if disturbance_at == step and state["on"]:
            knocked, support = random.choice(list(state["on"].items()))
            state["on"].pop(knocked); state["ontable"].add(knocked)
            target = goal["on"].get(knocked)
            msg = f"Reactive layer: {knocked} knocked over from {support}; repairing current plan locally."
            interventions.append(msg); trace.append(msg)
            if target:
                if state["holding"] is not None and state["holding"] != knocked:
                    held = state["holding"]
                    state = apply(state, Action("PUTDOWN", held))
                    trace.append(f"Reactive repair: PUTDOWN({held}) to free the gripper")
                if state["holding"] is None:
                    state = apply(state, Action("PICKUP", knocked))
                    trace.append(f"Reactive repair: PICKUP({knocked})")
                if state["holding"] == knocked and clear(state, target, boxes_in(state, goal)):
                    state = apply(state, Action("STACK", knocked, target))
                    trace.append(f"Reactive repair: STACK({knocked},{target})")

            elif knocked in goal["ontable"]:
                state = apply(state, Action("PICKUP", knocked))
                state = apply(state, Action("PUTDOWN", knocked))
                trace.append(f"Reactive repair: PUTDOWN({knocked})")

        if action.name == "UNSTACK" and state["on"].get(action.box) != action.target:
            trace.append(f"Step {step}: skipped {action}; disturbance changed its support")
            index += 1; continue
        if action.name == "PICKUP" and action.box not in state["ontable"]:
            trace.append(f"Step {step}: skipped {action}; box is no longer on table")
            index += 1; continue
        if action.name == "PUTDOWN" and state["holding"] != action.box:
            trace.append(f"Step {step}: skipped {action}; robot is not holding that box")
            index += 1; continue
        if action.name == "STACK":
            if state["holding"] != action.box or not clear(state, action.target, boxes_in(state, goal)):
                trace.append(f"Step {step}: skipped {action}; immediate preconditions no longer hold")
                index += 1; continue
        state = apply(state, action); trace.append(f"Step {step}: {action}"); index += 1
    return trace, interventions, state

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def demo():
    cases = [
        ({"on": {"B":"A","D":"C","F":"E"},"ontable":["A","C","E","G"]}, {"on":{"A":"B","C":"D","E":"F"},"ontable":["B","D","F","G"]}),
        ({"on": {"C":"B","E":"D"},"ontable":["A","B","D","F","G"]}, {"on":{"B":"A","D":"C","F":"E"},"ontable":["A","C","E","G"]}),
    ]
    for number, (start, goal) in enumerate(cases, 1):
        plan = gsp_plan(start, goal); seq, layers = nonlinear_plan(start, goal)
        print(f"\nPlanning test case {number}")
        print("Goal Stack Plan:", " -> ".join(map(str, plan)))
        print("Nonlinear layers:", " | ".join(", ".join(map(str, x)) for x in layers))
        trace, interventions, final_state = execute_with_reactive(start, plan, goal=goal, disturbance_at=max(2, len(plan)//2))
        print("Reactive interventions:", len(interventions))
        print("Goal reached after reactive execution:", goal_reached(final_state, goal))

def goal_reached(s, goal):
    s = normalize(s); goal = normalize(goal)
    return s["on"] == goal["on"] and s["ontable"] == goal["ontable"] and s["holding"] is None

def main():
    if len(sys.argv) == 3:
        start, goal = load_json(sys.argv[1]), load_json(sys.argv[2])
        plan = gsp_plan(start, goal)
        print("Goal Stack Plan")
        for i, action in enumerate(plan, 1): print(f"{i}. {action}")
        trace, interventions, final_state = execute_with_reactive(start, plan, goal=goal, disturbance_at=max(2, len(plan)//2))
        print("\nExecution trace")
        print("\n".join(trace))
        print(f"\nGoal reached after reactive execution: {goal_reached(final_state, goal)}")
        for item in interventions: print(item)

    else:
        demo()

if __name__ == "__main__": main()
