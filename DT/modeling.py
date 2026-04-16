import os
from pprint import pprint
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import simpy

from DT.components.Machine import Machine_pool
from DT.components.Monitor import Monitor
from DT.components.Process import Process
from DT.components.Sink import Sink
from DT.components.Source import Source
from DT.step_runner import DecisionManager, run_env_with_agent
from DT.utils.benchmarking_converter import convert_benchmarking_data
from DT.utils.postprocessing import plot_gantt_chart


PROBLEM_RULES: Dict[str, Tuple[str, str, str]] = {
    "JSSP": ("FSPT", "FIFO", "SPT"),
    "PFSP": ("JOHNSON", "FIFO", "FIFO"),
    "PMSP": ("WSPT", "WSPT", "WSPT"),
}


def get_rules(problem_type: str) -> Tuple[str, str, str]:
    return PROBLEM_RULES.get(problem_type, ("RANDOM", "RANDOM", "RANDOM"))


def load_data_from_unified_csv(data_dir: str, problem_name: str) -> Optional[Dict]:
    print(f"\n--- STEP 2: '{problem_name}' unified CSV loading start ---")
    try:
        csv_path = os.path.join(data_dir, f"problem_{problem_name}.csv")
        df = pd.read_csv(csv_path)

        job_info, op_info, machine_info = {}, {}, {}

        for machine_id, group in df.groupby("machine", sort=False):
            machine_id_str = str(machine_id)
            unique_processes = group["process"].unique().tolist()
            machine_info[machine_id_str] = {
                "id": machine_id_str,
                "capacity": int(group["capacity"].iloc[0]),
                "processes": [str(p) for p in unique_processes],
            }

        for op_id, group in df.groupby(
            df["job"].astype(str) + "_" + df["operation"].astype(str),
            sort=False,
        ):
            op_id_str = str(op_id)
            op_info[op_id_str] = {
                "id": op_id_str,
                "process": group["process"].tolist(),
                "machine": [str(m) for m in group["machine"].tolist()],
                "processing_time": group["processing_time"].tolist(),
            }

        def get_op_seq_num(op_name: str) -> int:
            return int(str(op_name).split("O")[-1])

        for job_id, group in df.groupby("job", sort=False):
            job_id_str = str(job_id)
            job_operations = (
                df.loc[df["job"] == job_id, "job"].astype(str)
                + "_"
                + df.loc[df["job"] == job_id, "operation"].astype(str)
            ).unique()
            sorted_ops = sorted(job_operations, key=get_op_seq_num)
            job_info[job_id_str] = {
                "id": job_id_str,
                "arrival_time": group["arrival_time"].iloc[0],
                "operations": [str(op) for op in sorted_ops],
            }
            if "weight" in group.columns and not pd.isna(group["weight"].iloc[0]):
                job_info[job_id_str]["weight"] = group["weight"].iloc[0]

        process_list = df["process"].unique().tolist()
        print(f"'{problem_name}' unified CSV loaded successfully.")
        return {
            "instance_name": problem_name,
            "job_info": job_info,
            "operation_info": op_info,
            "machine_info": machine_info,
            "process_list": process_list,
        }
    except FileNotFoundError as exc:
        print(f"Error: unified CSV file not found: {exc}")
        return None
    except Exception as exc:
        print(f"Error while parsing data: {exc}")
        return None


def build_simulation(
    problem_data: Dict,
    event_log_path: str,
    sequencing_rule: str,
    routing_rule: str,
    dispatching_rule: str,
    significant_digits: int,
    decision_policy=None,
    agent=None,
):
    env = simpy.Environment()
    model = {}
    resource = {}
    monitor = Monitor(event_log_path, significant_digits)
    decision_manager = DecisionManager(env=env, agent=agent) if agent is not None else None

    resource["Machine_pool"] = Machine_pool(
        monitor,
        problem_data,
        env,
        dispatching_rule,
        decision_policy=decision_policy,
        decision_manager=decision_manager,
    )

    model["Source"] = Source(
        model,
        monitor,
        "Source",
        problem_data,
        env,
        sequencing_rule,
        routing_rule,
        decision_policy=decision_policy,
        decision_manager=decision_manager,
    )
    model["Sink"] = Sink(model, monitor, "Sink", env)

    for proc_id in problem_data.get("process_list", []):
        model[proc_id] = Process(
            model,
            resource,
            monitor,
            proc_id,
            problem_data,
            env,
            dispatching_rule,
            routing_rule,
            decision_policy=decision_policy,
            decision_manager=decision_manager,
        )

    if decision_policy is not None and hasattr(decision_policy, "bind_runtime"):
        decision_policy.bind_runtime(env=env, monitor=monitor, model=model, resource=resource)
    if decision_manager is not None:
        decision_manager.bind_runtime(env=env, monitor=monitor, model=model, resource=resource)
        decision_manager.problem_data = problem_data

    return env, model, resource, monitor, decision_manager


def run_simulation(
    problem_data: Dict,
    event_log_path: str,
    sequencing_rule: str,
    routing_rule: str,
    dispatching_rule: str,
    significant_digits: int,
    decision_policy=None,
) -> Monitor:
    print(
        f"\n--- STEP 3: simulation start "
        f"(Seq={sequencing_rule}, Route={routing_rule}, Dispatch={dispatching_rule}) ---"
    )
    env, _, _, monitor, _ = build_simulation(
        problem_data=problem_data,
        event_log_path=event_log_path,
        sequencing_rule=sequencing_rule,
        routing_rule=routing_rule,
        dispatching_rule=dispatching_rule,
        significant_digits=significant_digits,
        decision_policy=decision_policy,
    )
    env.run()
    print("Simulation finished.")
    return monitor


def run_simulation_stepwise(
    problem_data: Dict,
    event_log_path: str,
    sequencing_rule: str,
    routing_rule: str,
    dispatching_rule: str,
    significant_digits: int,
    agent,
) -> Monitor:
    print(
        f"\n--- STEP 3: stepwise simulation start "
        f"(Seq={sequencing_rule}, Route={routing_rule}, Dispatch={dispatching_rule}) ---"
    )
    env, _, _, monitor, decision_manager = build_simulation(
        problem_data=problem_data,
        event_log_path=event_log_path,
        sequencing_rule=sequencing_rule,
        routing_rule=routing_rule,
        dispatching_rule=dispatching_rule,
        significant_digits=significant_digits,
        agent=agent,
    )
    run_env_with_agent(env, decision_manager)
    print("Simulation finished.")
    return monitor


def get_problem_names(folder_path: str):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid folder path: {folder_path}")

    file_names = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            name, _ = os.path.splitext(file)
            file_names.append(name)
    return file_names


def parse_jssp_results(result_path: str):
    df = pd.read_csv(result_path)
    return {row["Instance"].lower(): {"spt": row["SPT Solution"]} for _, row in df.iterrows()}


def parse_pfsp_optimums(data_path: str):
    df = pd.read_csv(data_path)
    return {
        row["Name"].lower(): {
            "n": row["n"],
            "m": row["m"],
            "LB": row["LB"],
            "UB": row["UB"],
            "Optimal": row["Optimal"],
            "UBFoundBy": row["UBFoundBy"],
            "Permutation": row["Permutation"],
        }
        for _, row in df.iterrows()
    }


def parse_pmsp_optimums(data_path: str):
    df = pd.read_csv(data_path)
    return dict(zip(df["name"], df["OFV"]))


def _normalize_events(completion_events: Sequence[str] | str) -> list[str]:
    if isinstance(completion_events, str):
        return [completion_events.lower()]
    return [event.lower() for event in completion_events]


def get_ofv_time(
    event_df: pd.DataFrame,
    weights: dict,
    completion_events: Sequence[str] | str = ("job completed", "job transferred to sink"),
    fallback_event: str = "operation complete",
):
    df = event_df.copy()
    df["event"] = df["event"].astype(str).str.lower()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["part_id", "time"])

    normalized_events = _normalize_events(completion_events)
    fallback_event = fallback_event.lower()

    def _completion_time(group: pd.DataFrame):
        g_sorted = group.sort_values("time")
        matched_completion = g_sorted[g_sorted["event"].isin(normalized_events)]
        if not matched_completion.empty:
            return matched_completion["time"].max()

        matched_fallback = g_sorted[g_sorted["event"].eq(fallback_event)]
        if not matched_fallback.empty:
            return matched_fallback["time"].max()
        return g_sorted["time"].max()

    completion_by_job = (
        df.groupby("part_id", as_index=True).apply(_completion_time, include_groups=False).rename("Cj")
    )

    weights_series = pd.Series(weights, name="wj")
    summary = pd.DataFrame(completion_by_job).join(weights_series, how="left")
    summary["wj"] = summary["wj"].fillna(0)
    summary["contrib"] = summary["wj"] * summary["Cj"]
    return summary["contrib"].sum()


def run_all_problems():
    dt_folder_path = os.path.dirname(os.path.abspath(__file__))
    is_benchmarking = True
    significant_digits = 10

    problem_types = {"PMSP", "PFSP", "JSSP"}
    baseline_folder = "baseline"
    data_root = "data"
    data_folder = "preprocessed"
    problem_folder = "raw"
    results_folder = "results"

    data_dir = os.path.join(dt_folder_path, data_root, data_folder)
    baseline_dir = os.path.join(dt_folder_path, baseline_folder)
    results_dir = os.path.join(dt_folder_path, results_folder)

    jssp_spt_makespan = parse_jssp_results(os.path.join(baseline_dir, "JSSP_SPT_Solution.csv"))
    pmsp_optimums = parse_pmsp_optimums(os.path.join(baseline_dir, "PMSP_OFV_Table.csv"))
    pfsp_optimums = parse_pfsp_optimums(os.path.join(baseline_dir, "Taillard_UB_Schedules OBrunner.csv"))

    errors = {}
    df_makespans = []

    for problem_type in problem_types:
        problem_dir = os.path.join(dt_folder_path, data_root, problem_folder, problem_type)
        problem_names = get_problem_names(problem_dir)
        df_makespan = pd.DataFrame(columns=["problem name"])

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        for problem_name in problem_names:
            try:
                if is_benchmarking:
                    txt_file_path = os.path.join(problem_dir, f"{problem_name}.txt")
                    convert_benchmarking_data(txt_file_path, data_dir, problem_type)

                data_dict = load_data_from_unified_csv(data_dir, problem_name)
                if not data_dict:
                    errors[f"{problem_type}-{problem_name}"] = "Data Loading Failed"
                    continue

                log_output_path = os.path.join(results_dir, problem_type, f"{problem_name}_event_log.csv")
                sequencing_rule, routing_rule, dispatching_rule = get_rules(problem_type)
                monitor = run_simulation(
                    data_dict,
                    log_output_path,
                    sequencing_rule,
                    routing_rule,
                    dispatching_rule,
                    significant_digits,
                )
                monitor.make_event_tracer()
                monitor.save_event_tracer()

                problem_key = problem_name.lower()
                if problem_type == "JSSP":
                    makespan = {
                        "problem name": problem_name,
                        "time": monitor.event_tracer.tail(1)["time"].iloc[0],
                        "SPT": jssp_spt_makespan[problem_key]["spt"] if problem_key in jssp_spt_makespan else "",
                    }
                elif problem_type == "PFSP":
                    pfsp_ref = pfsp_optimums.get(problem_key, {})
                    makespan = {
                        "problem name": problem_name,
                        "time": monitor.event_tracer.tail(1)["time"].iloc[0],
                        "LB": pfsp_ref.get("LB", ""),
                        "UB": pfsp_ref.get("UB", ""),
                    }
                elif problem_type == "PMSP":
                    weights = {job_id: info["weight"] for job_id, info in data_dict["job_info"].items()}
                    ofv_time = get_ofv_time(
                        monitor.event_tracer,
                        weights,
                        completion_events=("job completed",),
                    )
                    makespan = {
                        "problem name": problem_name,
                        "time": str(int(ofv_time)),
                        "optimum": pmsp_optimums[problem_key] if problem_key in pmsp_optimums else "",
                        "LB": "",
                        "UB": "",
                    }
                else:
                    makespan = {"problem name": "", "time": "", "optimum": ""}

                df_makespan = pd.concat([df_makespan, pd.DataFrame([makespan])], ignore_index=True)
                plot_gantt_chart(log_output_path)
            except Exception as exc:
                errors[f"{problem_type}-{problem_name}"] = str(exc)
                continue

        df_makespans.append((problem_type, df_makespan))

    with pd.ExcelWriter(os.path.join(results_dir, "makespans.xlsx"), engine="openpyxl") as writer:
        for sheet_name, df in df_makespans:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    if errors:
        pprint(errors)


def main():
    dt_folder_path = os.path.dirname(os.path.abspath(__file__))
    is_benchmarking = True
    significant_digits = 10
    problem_type = "JSSP"
    problem_name = "la01"

    data_root = "data"
    data_folder = "preprocessed"
    problem_folder = "raw"
    results_folder = "results"

    data_dir = os.path.join(dt_folder_path, data_root, data_folder)
    problem_dir = os.path.join(dt_folder_path, data_root, problem_folder, problem_type)
    results_dir = os.path.join(dt_folder_path, results_folder)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if is_benchmarking:
        txt_file_path = os.path.join(problem_dir, f"{problem_name}.txt")
        convert_benchmarking_data(txt_file_path, data_dir, problem_type)

    data_dict = load_data_from_unified_csv(data_dir, problem_name)
    if not data_dict:
        print("Data loading failed.")
        return

    log_output_path = os.path.join(results_dir, f"event_log_{problem_name}.csv")
    sequencing_rule, routing_rule, dispatching_rule = get_rules(problem_type)

    monitor = run_simulation(
        data_dict,
        log_output_path,
        sequencing_rule,
        routing_rule,
        dispatching_rule,
        significant_digits,
    )

    print("\n--- STEP 4: save result ---")
    monitor.make_event_tracer()
    monitor.save_event_tracer()
    print(f"Saved final event log to '{log_output_path}'.")
    plot_gantt_chart(log_output_path)


if __name__ == "__main__":
    main()
