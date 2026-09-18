import argparse
import json
from pathlib import Path

RULE_FILE=Path(__file__).with_name("rules.json")

def load_rules(path=RULE_FILE):
    with open(path,encoding="utf-8") as f: return json.load(f)

def diagnose(symptoms,rules=None):
    rules=rules or load_rules(); facts=set(symptoms); conclusions=set(); trace=[]
    changed=True

    while changed:
        changed=False
        for rule in rules:
            if rule["then"] not in conclusions and set(rule["if_all"]).issubset(facts):
                conclusions.add(rule["then"]); facts.add(rule["then"]); trace.append((rule["id"],rule["then"],rule["message"])); changed=True

    return sorted(conclusions),trace

def add_rule(rule,path=RULE_FILE):
    rules=load_rules(path); rules.append(rule)
    with open(path,"w",encoding="utf-8") as f: json.dump(rules,f,indent=2)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--symptoms",default="wheel_slip,motor_current_high"); parser.add_argument("--add-rule",action="store_true"); parser.add_argument("--rule-id"); parser.add_argument("--if-all"); parser.add_argument("--then"); parser.add_argument("--message"); args=parser.parse_args()

    if args.add_rule:
        if not all([args.rule_id,args.if_all,args.then,args.message]): parser.error("--add-rule needs --rule-id, --if-all, --then and --message")
        add_rule({"id":args.rule_id,"if_all":[x.strip() for x in args.if_all.split(",")],"then":args.then,"message":args.message}); print("Rule added."); return

    diagnoses,trace=diagnose([x.strip() for x in args.symptoms.split(",") if x.strip()])

    print("Diagnosis:",", ".join(diagnoses) if diagnoses else "No matching fault")
    print("Why trace:")

    for rid,result,message in trace: print(f"{rid}: {result} - {message}")

if __name__ == "__main__": main()