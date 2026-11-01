import csv
import json
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()
client = genai.Client()

FAQ = open("data/faq.json", encoding="utf-8").read()
MODELS = ["gemini-flash-lite-latest"]
JSON_VERSIONS = ["v4", "v5", "v6"]

# usage: python evaluate.py tests/pilot_cases.csv v4 v5
TEST_FILE = sys.argv[1] if len(sys.argv) > 1 else "tests/test_cases.csv"
VERSIONS = sys.argv[2:] if len(sys.argv) > 2 else ["v4", "v5", "v6"]


class Reply(BaseModel):
    answer: str
    in_faq: bool
    language: str


with open(TEST_FILE, encoding="utf-8") as f:
    tests = list(csv.DictReader(f))


def ask(version, system, question):
    use_json = version in JSON_VERSIONS
    config = types.GenerateContentConfig(
        system_instruction=system + "\n\nFAQ:\n" + FAQ,
        temperature=0,
        response_mime_type="application/json" if use_json else None,
        response_schema=Reply if use_json else None,
    )
    for model in MODELS:
        for attempt in range(2):
            try:
                start = time.time()
                r = client.models.generate_content(
                    model=model, contents=question, config=config
                )
                text, json_ok, in_faq = r.text, None, None
                if use_json:
                    try:
                        data = json.loads(r.text)
                        text, in_faq, json_ok = data["answer"], int(data["in_faq"]), 1
                    except Exception:
                        json_ok = 0
                return text, time.time() - start, json_ok, in_faq
            except Exception as e:
                print("   ERROR:", str(e)[:150])
                time.sleep(5)
    return "ERROR", 0, (0 if use_json else None), None


def matches(answer, options_text):
    clean = answer.lower().replace("’", "'").replace(" ", "")
    options = [o.strip().lower().replace(" ", "") for o in options_text.split("|") if o.strip()]
    return any(o in clean for o in options)


rows = []
for v in VERSIONS:
    system = open(f"prompts/{v}.txt", encoding="utf-8").read()
    for t in tests:
        answer, secs, json_ok, in_faq = ask(v, system, t["question"])
        ok = matches(answer, t["expected_keyword"])
        better = (t.get("better_keyword") or "").strip()
        precise = int(matches(answer, better)) if better else None
        rows.append(
            {
                "version": v,
                "type": t["type"],
                "question": t["question"],
                "answer": answer,
                "correct": int(ok),
                "precise": precise,
                "json_ok": json_ok,
                "in_faq": in_faq,
                "seconds": round(secs, 2),
            }
        )
        print(v, "PASS" if ok else "FAIL", "-", t["question"])
        time.sleep(4)

name = TEST_FILE.split("/")[-1].replace(".csv", "")
out = f"results_{name}_{'_'.join(VERSIONS)}.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\n=== RESULTS ===", TEST_FILE)
for v in VERSIONS:
    r = [x for x in rows if x["version"] == v]
    print(f"{v}: accuracy {sum(x['correct'] for x in r) / len(r) * 100:.1f}%")
    p = [x for x in r if x["precise"] is not None]
    if p:
        print(f"{v}: precise wording {sum(x['precise'] for x in p) / len(p) * 100:.1f}% ({len(p)} questions)")
    if v in JSON_VERSIONS:
        print(f"{v}: valid JSON {sum(x['json_ok'] for x in r) / len(r) * 100:.1f}%")
        print(f"{v}: fully answered from FAQ {sum((x['in_faq'] or 0) for x in r) / len(r) * 100:.1f}%")
print("Saved:", out)