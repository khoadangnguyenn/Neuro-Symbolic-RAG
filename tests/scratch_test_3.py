import json
from exact_pipeline.engines.symbolic_solver import run_symbolic_solver

translation = {
    "predicates": [
        "high_quality_camera(x)",
        "long_battery_life(x)"
    ],
    "premises_fol": [
        "∀x (high_quality_camera(x) → long_battery_life(x))",
        "¬long_battery_life(drone_x)"
    ],
    "options_fol": [
        "high_quality_camera(drone_x)",
        "¬high_quality_camera(drone_x)"
    ]
}

semantics = {
    "query_type": "multiple_choice",
    "intent": "choose_true"
}

res = run_symbolic_solver(semantics, translation)
print(json.dumps(res, indent=2))
