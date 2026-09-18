import math
import pandas as pd

READINGS = [
    {"id":1,"weight_deviation":0.03,"crack_severity":0.02,"tilt":0},
    {"id":2,"weight_deviation":0.18,"crack_severity":0.10,"tilt":0},
    {"id":3,"weight_deviation":0.42,"crack_severity":0.70,"tilt":1},
    {"id":4,"weight_deviation":0.05,"crack_severity":0.82,"tilt":0},
    {"id":5,"weight_deviation":0.58,"crack_severity":0.15,"tilt":1},
    {"id":6,"weight_deviation":0.30,"crack_severity":0.45,"tilt":0},
    {"id":7,"weight_deviation":0.75,"crack_severity":0.78,"tilt":1},
    {"id":8,"weight_deviation":0.12,"crack_severity":0.04,"tilt":1},
    {"id":9,"weight_deviation":0.65,"crack_severity":0.35,"tilt":0},
    {"id":10,"weight_deviation":0.22,"crack_severity":0.65,"tilt":0},
]

def nonmonotonic_demo():
    base_facts = set()
    initial = "NotDamaged" if "damaged" not in base_facts else "Damaged"
    facts = {"tilt_sensor_triggered"}
    updated = "Damaged" if "tilt_sensor_triggered" in facts else "NotDamaged"
    classical = "Undetermined" if "damaged" not in facts else "Damaged"

    return initial, updated, classical

def likelihood(reading):
    w = reading["weight_deviation"]
    c = reading["crack_severity"]
    t = reading["tilt"]
    p_e_d = (0.25 + 0.65*w) * (0.20 + 0.75*c) * (0.25 if not t else 0.90)
    p_e_n = (0.85 - 0.45*w) * (0.90 - 0.50*c) * (0.80 if not t else 0.25)

    return max(p_e_d, 1e-9), max(p_e_n, 1e-9)

def bayes(reading, prior=0.20):
    p_d_e, p_n_e = likelihood(reading)
    return prior*p_d_e/(prior*p_d_e+(1-prior)*p_n_e)

def combine_cf(a, b):
    if a >= 0 and b >= 0:
        return a + b*(1-a)
    if a <= 0 and b <= 0:
        return a + b*(1+a)
    denom = 1 - min(abs(a), abs(b))

    return (a+b)/denom if denom else 0.0

def cf_sensor(reading):
    w = min(1.0, reading["weight_deviation"])
    c = min(1.0, reading["crack_severity"])
    t = 0.75 if reading["tilt"] else -0.15

    return 0.70*w + 0.85*c + 0.65*t - 0.20

def certainty_factor(reading):
    values = [0.75*reading["weight_deviation"]-0.10, 0.90*reading["crack_severity"]-0.10, 0.70 if reading["tilt"] else -0.15]
    combined = values[0]

    for value in values[1:]:
        combined = combine_cf(combined, value)
    parallel = combined
    serial = values[0]

    for value in values[1:]:
        serial = serial * max(0.0, value)
    return max(0.0, min(1.0, (parallel+1)/2)), max(0.0, min(1.0, serial))

def bn_fallback(reading):
    p_noise = 0.25
    p_weight_d = 0.10 + 0.75*reading["weight_deviation"]
    p_crack_d = 0.05 + 0.90*reading["crack_severity"]
    p_tilt_d = 0.15 + 0.70*reading["tilt"]

    return p_noise*0.35*p_weight_d + (1-p_noise)*0.65*p_weight_d if False else (0.25*p_weight_d + 0.45*p_crack_d + 0.30*p_tilt_d)

def bayesian_network(reading):
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination

        model = DiscreteBayesianNetwork([("SensorNoise","WeightReading"),("CrackVisible","Damaged"),("Tilt","Damaged")])

        cpd_noise = TabularCPD("SensorNoise", 2, [[0.75],[0.25]])

        cpd_crack = TabularCPD("CrackVisible", 2, [[1-reading["crack_severity"]],[reading["crack_severity"]]])

        cpd_tilt = TabularCPD("Tilt", 2, [[1-reading["tilt"]],[reading["tilt"]]])

        cpd_weight = TabularCPD("WeightReading", 2, [[0.85,0.35],[0.15,0.65]], evidence=["SensorNoise"], evidence_card=[2])

        cpd_damage = TabularCPD("Damaged", 2, [[0.95,0.55,0.60,0.10],[0.05,0.45,0.40,0.90]], evidence=["CrackVisible","Tilt"], evidence_card=[2,2])

        model.add_cpds(cpd_noise, cpd_crack, cpd_tilt, cpd_weight, cpd_damage)

        inference = VariableElimination(model)

        result = inference.query(["Damaged"], evidence={"CrackVisible": int(reading["crack_severity"] >= 0.5), "Tilt": int(reading["tilt"])})

        return float(result.values[1])
    except Exception:
        return bn_fallback(reading)

def ds_combine(m1, m2):
    out = {"D":0.0,"N":0.0,"U":0.0}
    conflict = 0.0

    for a, ma in m1.items():
        for b, mb in m2.items():
            if a == "D" and b == "D": key = "D"
            elif a == "N" and b == "N": key = "N"
            elif a == "U": key = b
            elif b == "U": key = a
            else: key = "X"
            mass = ma*mb
            if key == "X": conflict += mass
            else: out[key] += mass
    scale = 1-conflict

    return {k:(v/scale if scale else 0.0) for k,v in out.items()}

def dempster_shafer(reading):
    w = reading["weight_deviation"]
    c = reading["crack_severity"]
    t = reading["tilt"]
    masses = [
        {"D":0.15+0.55*w,"N":0.55-0.35*w,"U":0.30-0.20*w},
        {"D":0.10+0.70*c,"N":0.65-0.40*c,"U":0.25-0.30*c},
        {"D":0.65 if t else 0.08,"N":0.12 if t else 0.70,"U":0.23 if t else 0.22},
    ]
    result = masses[0]

    for mass in masses[1:]: result = ds_combine(result, mass)
    belief = result["D"]
    plausibility = result["D"] + result["U"]

    return belief, plausibility

def fuzzy_score(reading):
    try:
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        weight=ctrl.Antecedent(__import__("numpy").arange(0,1.01,0.01),"weight_deviation")
        crack=ctrl.Antecedent(__import__("numpy").arange(0,1.01,0.01),"crack_severity")
        damage=ctrl.Consequent(__import__("numpy").arange(0,1.01,0.01),"damage_score")

        weight["low"]=fuzz.trimf(weight.universe,[0,0,0.35]); weight["medium"]=fuzz.trimf(weight.universe,[0.2,0.5,0.8]); weight["high"]=fuzz.trimf(weight.universe,[0.6,1,1])
        crack["none"]=fuzz.trimf(crack.universe,[0,0,0.2]); crack["minor"]=fuzz.trimf(crack.universe,[0.1,0.4,0.7]); crack["major"]=fuzz.trimf(crack.universe,[0.5,1,1])
        damage["low"]=fuzz.trimf(damage.universe,[0,0,0.35]); damage["medium"]=fuzz.trimf(damage.universe,[0.2,0.5,0.8]); damage["high"]=fuzz.trimf(damage.universe,[0.65,1,1])

        rules=[ctrl.Rule(weight["low"] & crack["none"],damage["low"]),ctrl.Rule(weight["low"] & crack["minor"],damage["medium"]),ctrl.Rule(weight["medium"] & crack["minor"],damage["medium"]),ctrl.Rule(weight["high"] | crack["major"],damage["high"]),ctrl.Rule(weight["medium"] & crack["major"],damage["high"])]
        system=ctrl.ControlSystem(rules); simulation=ctrl.ControlSystemSimulation(system)
        simulation.input["weight_deviation"]=reading["weight_deviation"]; simulation.input["crack_severity"]=reading["crack_severity"]; simulation.compute()
        score=float(simulation.output["damage_score"])

        if reading["tilt"]: score=min(1.0,score+0.1)

        return score

    except Exception:
        w=reading["weight_deviation"]; c=reading["crack_severity"]
        score=0.55*w+0.65*c+0.1*reading["tilt"]

        return max(0.0,min(1.0,score))

def compare():
    rows=[]
    for r in READINGS:
        cf, serial = certainty_factor(r)
        belief, plausibility = dempster_shafer(r)
        rows.append({"Case":r["id"],"Nonmonotonic":"Damaged" if (r["tilt"] or r["crack_severity"]>0.8) else "NotDamaged","Bayes":round(bayes(r),3),"CF":round(cf,3),"CF_serial":round(serial,3),"BN":round(bayesian_network(r),3),"DS belief":round(belief,3),"DS interval":f"[{belief:.2f}, {plausibility:.2f}]","Fuzzy":round(fuzzy_score(r),3)})

    return pd.DataFrame(rows)

def main():
    print("Nonmonotonic demo:", nonmonotonic_demo())
    df=compare()
    print(df.to_string(index=False))
    disagreements=[]

    for _, row in df.iterrows():
        vals=[row["Bayes"]>=0.5,row["CF"]>=0.5,row["BN"]>=0.5,row["Fuzzy"]>=0.5]
        if len(set(vals))>1: disagreements.append(int(row["Case"]))
    print("Disagreement cases:", disagreements[:5])

if __name__ == "__main__": main()