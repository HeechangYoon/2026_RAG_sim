from typing import List, Dict
from .Operation import Operation


class Job:
    """
    하나의 '작업(Job)'에 대한 모든 정보와 현재 진행 상태를 관리하는 데이터 클래스.
    """

    def __init__(self, job_info: Dict, operations_data: Dict):
        self.id: str = job_info['id']
        self.arrival_time: float = job_info['arrival_time']
        if "weight" in job_info:
            self.weight = job_info['weight']

        self.operation_list: List[Operation] = [
            Operation(operations_data[op_id]) for op_id in job_info['operations']
        ]

        assigned_keys = {'id', 'arrival_time', 'operations'}
        for key, value in job_info.items():
            if key not in assigned_keys:
                setattr(self, key, value)

        self.status = "departed"

        self.current_process = None
        self.step: int = 0  # 현재 진행 중인 operation_list의 인덱스
        self.completion_time: float = -1.0  # 완료 시간 (-1은 미완료 의미)
        self.last_operation_ready_time: float = float(self.arrival_time)
        self.last_process_entry_time: float = float(self.arrival_time)
        self.last_machine_queue_entry_time: float = float(self.arrival_time)
        self.last_dispatch_start_time: float = -1.0

    @property
    def current_operation(self) -> Operation:
        """현재 수행해야 할 Operation 객체를 반환합니다."""
        if not self.is_completed():
            return self.operation_list[self.step]
        return None

    def complete_step(self):
        """현재 단계를 완료하고 다음 단계로 넘어갑니다."""
        self.step += 1

    def is_completed(self) -> bool:
        """Job의 모든 공정이 완료되었는지 확인합니다."""
        return self.step >= len(self.operation_list)

    def __repr__(self) -> str:
        """Job 객체를 출력할 때의 형식을 정의합니다."""
        return f"Job(id={self.id}, step={self.step}/{len(self.operation_list)})"
