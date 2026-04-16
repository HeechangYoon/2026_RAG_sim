import pandas as pd
from io import StringIO


class Monitor:
    """
    시뮬레이션 동안 발생하는 모든 이벤트를 기록하고,
    최종적으로 CSV 파일로 저장하는 역할을 합니다.
    """

    def __init__(self, event_log_path, significant_digits):
        self.event_log_path = event_log_path
        self.significant_digits = significant_digits
        self.log_buffer = StringIO()
        self.HEADERS = ["time", "part_id", "operation", "process", "machine", "event"]
        self.log_buffer.write(",".join(self.HEADERS) + "\n")
        self.event_tracer = None

        self.job_list = []

    def record(self, time, part_id=None, operation=None, process=None, machine=None, event=None):
        """하나의 이벤트 로그를 버퍼에 기록합니다."""
        # 이벤트 이름도 저장 시 소문자로 통일
        event_str = str(event).lower() if event else ''

        log_entry = (
            f"{round(time, self.significant_digits)},"
            f"{part_id or ''},"
            f"{operation or ''},"
            f"{process or ''},"
            f"{machine or ''},"
            f"{event_str}\n"
        )
        self.log_buffer.write(log_entry)

    def make_event_tracer(self):
        """메모리 버퍼에 저장된 로그를 pandas DataFrame으로 변환합니다."""
        self.log_buffer.seek(0)
        self.event_tracer = pd.read_csv(self.log_buffer)

    def save_event_tracer(self):
        """DataFrame으로 변환된 이벤트 로그를 실제 CSV 파일로 저장합니다."""
        if self.event_tracer is not None:
            self.event_tracer.to_csv(self.event_log_path, index=False)
        else:
            print("경고: 저장할 이벤트 트레이서가 생성되지 않았습니다.")