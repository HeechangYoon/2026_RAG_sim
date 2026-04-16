import os
import csv
from typing import List, Dict


def convert_benchmarking_data(txt_file_path: str, output_dir: str, problem_type: str):
    """
    문제 유형에 따라 .txt 벤치마크 데이터를 파싱하여 하나의 통합된 CSV 파일로 변환합니다.
    """
    print(f"\n--- STEP 1: {problem_type} 유형의 .txt 파일을 단일 CSV로 변환 시작 ---")
    try:
        if not os.path.exists(txt_file_path):
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {txt_file_path}")

        if problem_type == "JSSP":
            rows = _parse_jssp_to_rows(txt_file_path)
        elif problem_type == "PFSP":
            rows = _parse_pfsp_to_rows(txt_file_path)
        elif problem_type == "PMSP":
            rows = _parse_pmsp_to_rows(txt_file_path)
        else:
            raise ValueError("올바르지 않은 문제 유형입니다.")

        problem_name = os.path.basename(txt_file_path).split('.')[0]
        output_path = os.path.join(output_dir, f"problem_{problem_name}.csv")
        _write_single_csv(output_path, rows)

        print(f"통합 CSV 파일 변환 성공. 결과가 '{output_path}'에 저장되었습니다.")

    except Exception as e:
        print(f"오류: .txt 파일 변환 실패: {e}")


def _write_single_csv(filepath: str, data: List[Dict]):
    """딕셔셔너리 리스트를 하나의 CSV 파일로 저장합니다. (생성된 순서 그대로 사용)"""
    if not data:
        print(f"경고: '{filepath}'에 쓸 데이터가 없습니다.")
        return

    # 데이터의 첫 번째 행(row)에 정의된 키 순서를 그대로 사용하여 헤더를 만듭니다.
    fieldnames = data[0].keys()

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def _parse_pmsp_to_rows(txt_file_path: str) -> List[Dict]:
    """PMSP .txt 파일을 파싱하여 요청된 컬럼 순서대로 행(row) 리스트를 생성합니다."""
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    header_pij = next(i for i, ln in enumerate(lines) if ln.lower().startswith('pij'))
    header_wj = next(i for i, ln in enumerate(lines) if ln.lower().startswith('wj'))

    machines_raw = [list(map(int, lines[i].split())) for i in range(header_pij + 1, header_wj)]
    weights_raw = list(map(int, lines[header_wj + 1].split()))

    num_machines, num_jobs = len(machines_raw), len(machines_raw[0])

    rows = []
    for j in range(num_jobs):
        for i in range(num_machines):
            job_id, machine_id = f"J{j + 1}", f"M{i + 1}"
            row = {
                'arrival_time': 0.0,
                'job': job_id,
                'operation': "O1",
                'process': "P1",
                'machine': machine_id,
                'capacity': 1,
                'processing_time': float(machines_raw[i][j]),
                'weight': float(weights_raw[j])
            }
            rows.append(row)
    return rows


def _parse_jssp_to_rows(txt_file_path: str) -> List[Dict]:
    """JSSP .txt 파일을 파싱하여 요청된 컬럼 순서대로 행(row) 리스트를 생성합니다."""
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    n_jobs, n_machines = map(int, lines[0].split())

    job_rows = [list(map(int, line.split())) for line in lines[1:n_jobs + 1]]
    if job_rows and all(len(row) == 2 * n_machines for row in job_rows):
        machines_raw = [row[0::2] for row in job_rows]
        times_raw = [row[1::2] for row in job_rows]
    else:
        times_raw = job_rows
        machines_raw = [list(map(int, line.split())) for line in lines[n_jobs + 1:n_jobs * 2 + 1]]

    rows = []
    for i in range(n_jobs):
        for j in range(n_machines):
            job_id, machine_id_num = f"J{i + 1}", machines_raw[i][j]
            proc_time, machine_id = times_raw[i][j], f"M{machine_id_num}"
            row = {
                'arrival_time': 0.0,
                'job': job_id,
                'operation': f"O{j + 1}",
                'process': f"P{j + 1}",
                'machine': machine_id,
                'capacity': 1,
                'processing_time': float(proc_time)
            }
            rows.append(row)
    return rows


def _parse_pfsp_to_rows(txt_file_path: str) -> List[Dict]:
    """PFSP .txt 파일을 파싱하여 요청된 컬럼 순서대로 행(row) 리스트를 생성합니다."""
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    header_parts = [int(part) for part in lines[0].split() if part.isdigit()]
    if len(header_parts) < 2: raise ValueError("PFSP 파일 헤더 오류")
    n_jobs, n_machines = header_parts[0], header_parts[1]

    data_lines = [line for line in lines[1:] if line and line[0].isdigit()]

    times_by_job = [list(map(int, line.split()))[1::2] for line in data_lines[:n_jobs]]

    rows = []
    for i in range(n_jobs):
        for j in range(n_machines):
            job_id, machine_id = f"J{i + 1}", f"M{j + 1}"
            if i < len(times_by_job) and j < len(times_by_job[i]):
                proc_time = times_by_job[i][j]
                row = {
                    'arrival_time': 0.0,
                    'job': job_id,
                    'operation': f"O{j + 1}",
                    'process': f"P{j + 1}",
                    'machine': machine_id,
                    'capacity': 1,
                    'processing_time': float(proc_time)
                }
                rows.append(row)
    return rows


def convert_banchmarking_data(txt_file_path: str, output_dir: str, problem_type: str):
    """Backward-compatible alias for the previous misspelled function name."""
    return convert_benchmarking_data(txt_file_path, output_dir, problem_type)
