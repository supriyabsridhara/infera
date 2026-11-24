from train_model import infer, load_model
import os

model = load_model(os.path.join("data", "bayesian_model.pkl"))

tests = [
    ("All OFF",             0, 0, 0),
    ("Door ON only",        1, 0, 0),
    ("Door+Motion ON",      1, 1, 0),
    ("All ON (intrusion)",  1, 1, 1),
]

print("Inference sanity checks:")
print(f"{'Scenario':<25} {'No Entry':>10} {'Authorized':>12} {'Intrusion':>12} {'Risk':<10}")
print("-" * 72)
for name, d, m, g in tests:
    r = infer(model, door=d, motion=m, glass=g)
    print(
        f"{name:<25} {r['no_entry']*100:>9.1f}%"
        f" {r['authorized']*100:>11.1f}%"
        f" {r['intrusion']*100:>11.1f}%  {r['risk_level']}"
    )
print("\nAll checks passed!")
