from rag_engine import build_rag, evaluate_retrieval

v = build_rag()
r, acc = evaluate_retrieval(v)

for x in r:
    status = '✅' if x['passed'] else '❌'
    print(f"{status} [{x['similarity']}] {x['question'][:50]}")
    if not x['passed']:
        print(f"   Expected: {x['expected_category']} | Found: {x['found_category']}")

print(f"\nAccuracy: {acc}%")