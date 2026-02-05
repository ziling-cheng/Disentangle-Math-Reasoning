import random
import pandas as pd
import json
import argparse

def generate_addsub_pairs(n_pairs=150, n_times=2, out_csv="addition_subtraction_sampled_names_and_numbers.csv", max_num=50, seed=42):
    random.seed(seed)

    names = [
        "James", "Emma", "William", "Olivia", "Benjamin", "Charlotte",
        "Henry", "Amelia", "Alexander", "Ava", "Samuel", "Sophia",
        "Jacob", "Mia", "Daniel", "Lily", "Michael", "Grace",
        "Ethan", "Ella", "Jack", "Chloe", "Lucas", "Harper",
        "Thomas", "Zoe", "Matthew", "Nora", "Nathan", "Isla"
    ]

    generated_pairs = set()
    rows = []

    for pair_id in range(n_pairs):  # one "template pair"
        name = random.choice(names)

        for _ in range(2 * n_times):  # 4 samples when n_times=2
            while True:
                # ensure a>b & a, b, a-b different
                a = random.randint(2, max_num)
                b = random.randint(1, a - 1)
                if len({a, b, a - b}) == 3 and (a, b) not in generated_pairs:
                    generated_pairs.add((a, b))
                    break

            rows.append({
                "name": name,
                "a": a,
                "b": b,
                "addition": a + b,
                "subtraction": a - b,
            })
            
    return pd.DataFrame(rows).to_csv(out_csv, index=False)


def generate_divisible_pairs(n_pairs=150, max_num=50, n_times=2, out_csv="multiplication_division_sampled_names_and_numbers.csv", seed=42):
    random.seed(seed)

    names = [
        "James", "Emma", "William", "Olivia", "Benjamin", "Charlotte",
        "Henry", "Amelia", "Alexander", "Ava", "Samuel", "Sophia",
        "Jacob", "Mia", "Daniel", "Lily", "Michael", "Grace",
        "Ethan", "Ella", "Jack", "Chloe", "Lucas", "Harper",
        "Thomas", "Zoe", "Matthew", "Nora", "Nathan", "Isla"
    ]

    # Precompute all valid (x, y) pairs where x is divisible by y
    all_pairs = [
        (x, y) for x in range(1, max_num + 1)
        for y in range(1, x + 1)
        if x % y == 0 and y != 1  # Exclude division by 1
    ]
    print(f"Total valid divisible (x, y) pairs: {len(all_pairs)}")

    total_required = n_pairs * n_times # clean and corrupted

    if total_required > len(all_pairs):
        print(f"Warning: requesting {total_required} pairs, but only {len(all_pairs)} available. Reuse will occur.")
        allow_replacement = True
    else:
        allow_replacement = False

    rows = []

    for _ in range(n_pairs):
        name = random.choice(names)
        
        for _ in range(n_times):
            # Get clean pair
            x1, y1 = random.choice(all_pairs) if allow_replacement else all_pairs.pop()
            
            # Find corrupted pair that meets the constraints
            max_attempts = 100  # Prevent infinite loops
            for _ in range(max_attempts):
                x2, y2 = random.choice(all_pairs) if allow_replacement else all_pairs.pop()
                if (x1 // y1 != x2 // y2) and (x1 * y1 != x2 * y2):
                    break
            else:
                # If no suitable pair found after max_attempts, just take any pair
                x2, y2 = random.choice(all_pairs) if allow_replacement else all_pairs.pop()
                print(f"Warning: Could not find pair meeting constraints for ({x1}, {y1})")
            
            
            rows.append({"name": name, "a": x1, "b": y1, "multiplication": x1*y1, "division":int(x1/y1)})
            rows.append({"name": name, "a": x2, "b": y2, "multiplication": x2*y2, "division":int(x2/y2)})
    
    return pd.DataFrame(rows).to_csv(out_csv, index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem_type",
        type=str,
        default="addition_subtraction",
        choices=["addition_subtraction", "multiplication_division"],
        help="Which paired operation templates to use.",
    )
    args = parser.parse_args()
    
    if args.problem_type == "addition_subtraction":
        generate_addsub_pairs()
    else:
        generate_divisible_pairs()