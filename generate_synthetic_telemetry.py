import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define the 6 Archetypes and their probability weights
ARCHETYPES = [
    "Independent Solver",
    "Structured Collaborator",
    "Prompt Engineer Solver",
    "Iterative Debugger",
    "AI-Dependent Constructor",
    "Blind Copier",
    "Exploratory Learner"
]
ARCHETYPE_WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.15, 0.05, 0.05]

def generate_telemetry_features(archetype, noise_level=0.1):
    """
    Generates a 51-feature vector for a single candidate based on their archetype.
    Uses overlapping normal distributions to prevent perfect synthetic separation.
    """
    row = {"archetype": archetype}
    
    # helper variable: how 'good' the coder is vs how reliant on AI
    is_independent = archetype in ["Independent Solver", "Iterative Debugger", "Exploratory Learner"]
    is_dependent = archetype in ["AI-Dependent Constructor", "Blind Copier"]
    is_collaborator = archetype in ["Structured Collaborator", "Prompt Engineer Solver"]
    
    # -------------------------------------------------------------
    # 1. IDE Interaction Features (10 signals)
    # -------------------------------------------------------------
    base_inserts = np.random.normal(loc=1500 if is_independent else 800 if is_collaborator else 200, scale=300)
    row["total_insert_events"] = max(10, int(base_inserts))
    
    base_deletes = np.random.normal(loc=row["total_insert_events"] * 0.4 if not is_dependent else 50, scale=50)
    row["total_delete_events"] = max(0, int(base_deletes))
    
    row["total_copy_events"] = max(0, int(np.random.normal(loc=15 if is_collaborator else 5, scale=5)))
    
    base_pastes = np.random.normal(loc=5 if is_independent else 20 if is_collaborator else 50, scale=10)
    row["total_paste_events"] = max(0, int(base_pastes))
    
    # High paste length for blind copiers
    avg_paste_len = np.random.normal(loc=15 if is_independent else 60 if is_collaborator else 350 if archetype == "Blind Copier" else 150, scale=30)
    row["avg_paste_length"] = max(0, int(avg_paste_len))
    
    row["max_paste_length"] = int(row["avg_paste_length"] * np.random.uniform(1.2, 3.0))
    # FIX: Simulate massive single-dump copies for Blind Copiers
    if archetype == "Blind Copier":
        row["max_paste_length"] = int(row["max_paste_length"] * np.random.uniform(2.0, 5.0))
        
    row["paste_to_insert_ratio"] = round(row["total_paste_events"] / (row["total_insert_events"] + 1), 4)
    
    row["undo_events"] = max(0, int(np.random.normal(loc=20 if is_independent else 5, scale=5)))
    row["redo_events"] = max(0, int(row["undo_events"] * np.random.uniform(0.1, 0.4)))
    
    base_runs = np.random.normal(loc=35 if archetype == "Iterative Debugger" else 15 if is_independent else 5, scale=4)
    row["run_compile_events"] = max(1, int(base_runs))

    # -------------------------------------------------------------
    # 2. Keystroke & Typing Dynamics (9 signals)
    # -------------------------------------------------------------
    base_wpm = np.random.normal(loc=85 if is_independent else 60 if is_collaborator else 25, scale=15)
    row["avg_typing_speed"] = max(5, round(base_wpm, 1))
    
    row["typing_burst_count"] = max(5, int(row["total_insert_events"] / np.random.uniform(15, 40)))
    row["typing_pause_frequency"] = max(2, int(row["typing_burst_count"] * np.random.uniform(0.5, 1.5)))
    
    # Dependent coders pause longer (waiting for AI)
    row["avg_pause_duration"] = round(np.random.normal(loc=5 if is_independent else 12 if is_collaborator else 25, scale=5), 1)
    row["longest_pause"] = round(row["avg_pause_duration"] * np.random.uniform(2, 6), 1)
    
    row["key_latency_mean"] = round(1000 / (row["avg_typing_speed"] * 5) * 60, 1) # ms per key roughly
    row["key_latency_variance"] = round(row["key_latency_mean"] * np.random.uniform(0.2, 0.8), 1)
    
    # Paste after idle
    row["paste_after_idle_time"] = round(np.random.normal(loc=3 if is_independent else 15 if is_collaborator else 35, scale=8), 1)
    
    # FIX: Clamp the ratio to prevent extreme mathematical outliers (division by near-zero) in clustering
    row["typing_vs_paste_ratio"] = min(10.0, round((row["total_insert_events"] * 5) / (row["total_paste_events"] * row["avg_paste_length"] + 1), 3))

    # -------------------------------------------------------------
    # 3. Code Evolution Features (10 signals)
    # -------------------------------------------------------------
    row["code_edit_distance"] = round(np.random.uniform(0.2, 0.8) if is_independent else np.random.uniform(0.1, 0.4), 3)
    row["total_code_versions"] = max(1, int(row["run_compile_events"] * np.random.uniform(0.8, 1.5)))
    row["avg_lines_added_per_revision"] = max(1, int(np.random.normal(loc=5 if is_independent else 20, scale=4)))
    row["avg_lines_removed_per_revision"] = max(0, int(row["avg_lines_added_per_revision"] * np.random.uniform(0.2, 0.9)))
    
    row["refactor_frequency"] = max(0, int(np.random.normal(loc=10 if is_independent else 2, scale=3)))
    row["compile_error_count"] = max(0, int(row["run_compile_events"] * np.random.uniform(0.1, 0.6)))
    
    # Dependent coders take longer to manually fix errors
    row["error_fix_time"] = round(np.random.normal(loc=20 if is_independent else 90 if is_dependent else 45, scale=20), 1)
    
    row["code_rewrite_ratio"] = round(np.random.uniform(0.3, 0.7) if is_independent else np.random.uniform(0.05, 0.2), 3)
    row["function_rewrite_count"] = max(0, int(np.random.normal(loc=4 if is_independent else 0, scale=1)))
    row["comment_addition_count"] = max(0, int(np.random.normal(loc=10 if is_collaborator else 2, scale=2)))

    # -------------------------------------------------------------
    # 4. AI Prompt Interaction Features (12 NLP signals)
    # -------------------------------------------------------------
    base_prompts = np.random.normal(loc=2 if is_independent else 12 if is_collaborator else 25 if is_dependent else 8, scale=3)
    row["total_prompts_sent"] = max(0, int(base_prompts))
    
    # Prompt engineers write long prompts
    row["avg_prompt_length"] = max(5, int(np.random.normal(loc=10 if is_independent else 60 if archetype=="Prompt Engineer Solver" else 20, scale=10)))
    row["max_prompt_length"] = int(row["avg_prompt_length"] * np.random.uniform(1.2, 3.0))
    
    # Blind copiers paste the exact prompt
    row["prompt_similarity_to_problem"] = round(np.random.normal(loc=0.1 if is_independent else 0.4 if is_collaborator else 0.9 if archetype=="Blind Copier" else 0.7, scale=0.1), 3)
    
    row["prompt_refinement_count"] = max(0, int(np.random.normal(loc=0 if is_independent else 5 if archetype in ["Prompt Engineer Solver", "Structured Collaborator"] else 1, scale=1)))
    row["prompt_entropy"] = round(np.random.uniform(0.7, 1.0) if archetype == "Exploratory Learner" else np.random.uniform(0.2, 0.6), 3)
    
    row["clarification_prompt_ratio"] = round(np.random.uniform(0.5, 0.9) if is_independent else np.random.uniform(0.1, 0.3), 3)
    
    # FIX: Ensure solution_request_ratio never drops below 0
    row["solution_request_ratio"] = max(0.0, round(1.0 - row["clarification_prompt_ratio"] - np.random.uniform(0, 0.1), 3))
    
    row["debugging_prompt_ratio"] = round(np.random.uniform(0.4, 0.8) if archetype == "Iterative Debugger" else np.random.uniform(0.1, 0.4), 3)
    
    row["ai_response_acceptance_rate"] = round(np.random.normal(loc=0.2 if is_independent else 0.6 if is_collaborator else 0.95 if archetype=="Blind Copier" else 0.8, scale=0.1), 3)
    row["ai_output_edit_distance"] = round(np.random.uniform(0.6, 0.9) if is_collaborator else np.random.uniform(0.01, 0.1) if is_dependent else np.random.uniform(0.2, 0.5), 3)
    row["prompt_to_code_latency"] = round(np.random.normal(loc=60 if is_collaborator else 10 if is_dependent else 30, scale=10), 1)

    # -------------------------------------------------------------
    # 5. Temporal Workflow Features (10 signals)
    # -------------------------------------------------------------
    base_duration = 3600 if archetype in ["Exploratory Learner", "Iterative Debugger"] else 1200 if is_dependent else 2700
    row["session_duration"] = max(600, int(np.random.normal(loc=base_duration, scale=400))) # 60m vs 45m vs 20m
    row["time_to_first_code"] = max(10, int(np.random.normal(loc=120 if is_independent else 300 if is_collaborator else 30, scale=30)))
    row["time_to_first_ai_prompt"] = max(10, int(np.random.normal(loc=1500 if is_independent else 300 if is_collaborator else 60, scale=100)))
    
    row["ai_usage_early_ratio"] = round(np.random.uniform(0, 0.2) if is_independent else np.random.uniform(0.6, 0.9) if is_dependent else np.random.uniform(0.3, 0.6), 3)
    row["ai_usage_late_ratio"] = round(np.random.uniform(0, 0.5) if is_independent else np.random.uniform(0.6, 0.9) if archetype=="Iterative Debugger" else np.random.uniform(0.2, 0.6), 3)
    
    row["iteration_cycle_count"] = max(1, int(row["session_duration"] / (row["time_to_first_code"] + 1) * np.random.uniform(0.1, 0.5)))
    row["avg_cycle_duration"] = round(row["session_duration"] / (row["iteration_cycle_count"] + 1), 1)
    
    row["focus_switch_count"] = max(0, int(row["total_prompts_sent"] * np.random.uniform(1.2, 2.5)))
    row["browser_to_editor_ratio"] = round(np.random.uniform(0.1, 0.3) if is_independent else np.random.uniform(0.5, 0.8) if is_dependent else np.random.uniform(0.3, 0.6), 3)
    row["idle_ratio"] = round(row["browser_to_editor_ratio"] * np.random.uniform(0.8, 1.2), 3)

    # -------------------------------------------------------------
    # Noise Injection (Defense Mechanism)
    # -------------------------------------------------------------
    # Inject 10% random noise to random features to prevent perfect synthetic separation
    for key in row.keys():
        if key != "archetype" and random.random() < noise_level:
            # Shift value wildly
            noise_multiplier = np.random.uniform(0.2, 5.0)
            row[key] = row[key] * noise_multiplier
            # Keep bounds reasonable for ratios
            if "ratio" in key or "rate" in key:
                row[key] = min(1.0, max(0.0, row[key]))
            elif isinstance(row[key], int):
                row[key] = int(row[key])
            else:
                row[key] = round(row[key], 3)
                
    return row

def main():
    print("Generating Synthetic Telemetry Dataset...")
    num_candidates = 2000
    noise_level = 0.15 # 15% random noise injection
    
    data = []
    for _ in range(num_candidates):
        # FIX: Use weighted choices for realistic interview distribution
        archetype = random.choices(ARCHETYPES, weights=ARCHETYPE_WEIGHTS, k=1)[0]
        candidate_data = generate_telemetry_features(archetype, noise_level=noise_level)
        data.append(candidate_data)
        
    df = pd.DataFrame(data)
    
    # Save the base dataset
    output_path = "c:/Users/hosan/Desktop/Research Project/synthetic_telemetry_51_signals.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully! ({num_candidates} rows, {len(df.columns)} columns)")
    print(f"Saved to: {output_path}")
    
    # Display sample and class balance
    print("\nArchetype Distribution:")
    print(df['archetype'].value_counts())
    
    print("\nSample Data (First 2 rows, 5 features):")
    print(df.iloc[:2, :6])
    
    # FIX: Dataset Validation Step
    print("\n--- Dataset Validation (Describe Ratios) ---")
    ratio_columns = [col for col in df.columns if 'ratio' in col or 'rate' in col][:5]
    print(df[ratio_columns].describe())

if __name__ == "__main__":
    main()
