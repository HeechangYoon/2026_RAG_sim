import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_gantt_chart(log_path: str):
    """
    event_log.csv 파일을 읽어 Matplotlib 기반의 간트 차트를 생성하고 저장합니다.
    (데이터 처리 및 오류 핸들링 로직을 최종 강화했습니다.)
    """
    print("\n--- STEP 5: Matplotlib 간트 차트 생성 시작 ---")

    try:
        df = pd.read_csv(log_path)

        # 1. 모든 컬럼 이름의 양쪽 공백을 제거하고 소문자로 통일합니다.
        df.columns = [col.strip().lower() for col in df.columns]

        # 2. 'event' 컬럼이 확실히 존재하는지 확인합니다.
        if 'event' not in df.columns:
            print(f"간트 차트 생성 실패: CSV 파일에 'event' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")
            return

        # 3. 데이터 처리를 위해 필요한 모든 컬럼이 있는지 확인합니다.
        required_cols = ['part_id', 'operation', 'process', 'machine', 'event', 'time']
        if not all(col in df.columns for col in required_cols):
            print(f"간트 차트 생성 실패: CSV에 필수 컬럼이 부족합니다. 필요: {required_cols}, 현재: {df.columns.tolist()}")
            return

        # pivot_table을 사용하여 시작/종료 시간 데이터 재구성
        pivot_df = df.pivot_table(
            index=['part_id', 'operation', 'process', 'machine'],
            columns='event',
            values='time'
        ).reset_index()

        # pivot_table이 만든 컬럼 이름도 소문자로 통일
        pivot_df.columns = [col.strip().lower() for col in pivot_df.columns]

        # 컬럼 이름 변경
        pivot_df.rename(columns={
            'job assigned': 'Start',
            'operation complete': 'Finish',
            'part_id': 'Job',
            'machine': 'Machine'
        }, inplace=True)

        # 'Start'와 'Finish' 컬럼이 모두 존재하는 데이터만 필터링
        if 'Start' not in pivot_df.columns or 'Finish' not in pivot_df.columns:
            print(f"오류: 'Start' 또는 'Finish' 컬럼을 생성하지 못했습니다. (event 로그 확인 필요)")
            print(f"Pivot 이후 컬럼: {pivot_df.columns.tolist()}")
            return

        gantt_df = pivot_df.dropna(subset=['Start', 'Finish']).copy()

        if gantt_df.empty:
            print("오류: 간트 차트를 그릴 데이터를 생성하지 못했습니다. (시작/종료 이벤트 쌍 매칭 실패)")
            return

    except Exception as e:
        print(f"간트 차트 데이터 처리 중 예상치 못한 오류 발생: {e}")
        return

    # (이하 간트 차트를 그리는 코드는 이전 답변과 동일)
    makespan = gantt_df['Finish'].max()
    unique_jobs = sorted(gantt_df['Job'].unique(), key=lambda x: int(x[1:]))

    colors = plt.cm.get_cmap('viridis', len(unique_jobs))
    job_color_map = {job: colors(i / (len(unique_jobs) - 1 if len(unique_jobs) > 1 else 1)) for i, job in
                     enumerate(unique_jobs)}

    num_machines = len(gantt_df['Machine'].unique())
    fig_height = max(8, num_machines * 0.7)
    fig, ax = plt.subplots(figsize=(25, fig_height))

    original_machine_names = sorted(gantt_df['Machine'].unique())
    sorted_machine_indices = sorted(range(len(original_machine_names)),
                                    key=lambda k: int(original_machine_names[k].split('M')[1].split('_')[0]))
    sorted_original_machine_names = [original_machine_names[i] for i in sorted_machine_indices]
    sorted_machine_display_names = [name.split('_')[0] for name in sorted_original_machine_names]

    y_ticks = np.arange(len(sorted_machine_display_names))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(sorted_machine_display_names)

    for i, machine_orig_name in enumerate(sorted_original_machine_names):
        machine_tasks = gantt_df[gantt_df['Machine'] == machine_orig_name]
        y_pos = y_ticks[i]

        for _, task in machine_tasks.iterrows():
            start_time = task['Start']
            duration = task['Finish'] - start_time
            job_name = task['Job']
            op_name = task['operation']
            op_short_name = str(op_name).split('_S')[-1] if '_S' in str(op_name) else str(op_name)

            bar_color = job_color_map.get(job_name, (0.5, 0.5, 0.5, 1.0))
            ax.barh(y_pos, duration, left=start_time, color=bar_color, edgecolor='black', height=0.6)

            color_rgb = bar_color[:3]
            text_color = 'white' if mcolors.rgb_to_hsv(color_rgb)[2] < 0.6 else 'black'

            text_content = f"{job_name}({op_short_name})"

            if duration > (makespan / 100):
                ax.text(start_time + duration / 2, y_pos, text_content, ha='center', va='center', color=text_color,
                        fontweight='bold', fontsize=8)

    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Machine', fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax.set_title('Gantt Chart for Job Shop Scheduling', fontsize=20)
    ax.grid(axis='x', linestyle='--')

    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=job_color_map[job]) for job in unique_jobs]
    ax.legend(legend_elements, unique_jobs, title="Jobs", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=18,
              title_fontsize=18)

    ax.text(1.0, 1.05, f"Makespan: {makespan:.2f}", transform=ax.transAxes, fontsize=18, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    results_dir = os.path.dirname(log_path)
    problem_name = os.path.basename(log_path).replace('event_log_', '').replace('.csv', '')
    chart_path = os.path.join(results_dir, f'gantt_chart_{problem_name}.png')

    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(chart_path, dpi=300)
    print(f"간트 차트가 이미지 파일로 '{chart_path}'에 저장되었습니다.")