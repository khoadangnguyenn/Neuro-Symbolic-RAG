from exact_pipeline.engines.symbolic_solver import run_symbolic_solver
import json

translation = {
    "predicates": [
        "completed_ethics_training(x)",
        "has_lab_access(x)",
        "handle_participant_data(x)",
        "has_supervisor_approval(x)",
        "join_study_alpha(x)",
        "listed_as_active_contributor(x)",
        "has_budget_approval(x)"
    ],
    "functions": [
        "enrolled_participants(x): Int"
    ],
    "premises_fol": [
        "∀x (completed_ethics_training(x) ∧ has_lab_access(x) → handle_participant_data(x))",
        "∀x (handle_participant_data(x) ∧ has_supervisor_approval(x) → join_study_alpha(x))",
        "∀x (join_study_alpha(x) → listed_as_active_contributor(x))",
        "completed_ethics_training(asha)",
        "has_lab_access(asha)",
        "has_supervisor_approval(asha)",
        "enrolled_participants(study_alpha) = 12"
    ],
    "condition_fol": "",
    "target_fol": "",
    "options_fol": [
        "join_study_alpha(asha)",
        "¬handle_participant_data(asha)",
        "has_budget_approval(asha)",
        "enrolled_participants(study_alpha) = 20"
    ]
}

semantics = {
    "query_type": "multiple_choice",
    "intent": "choose_true"
}

res = run_symbolic_solver(semantics, translation)
print(json.dumps(res, indent=2))
