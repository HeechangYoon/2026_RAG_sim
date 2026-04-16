import simpy


class Sink:
    """
    완료된 Job들을 수집하고, 시뮬레이션의 최종 결과를 계산 및 출력
    """

    def __init__(self, model, monitor, id, env: simpy.Environment):
        self.env = env
        self.model = model
        self.monitor = monitor
        self.id = id

        self.parts_rec = 0
        self.last_arrival = 0.0

        self.store = simpy.Store(env=self.env)

        self.env.process(self.run())

    def run(self):
        while True:
            job = yield self.store.get()
            job.status = 'finished'
            self.parts_rec += 1
            self.last_arrival = self.env.now
            self.monitor.record(time=self.env.now, part_id=job.id, operation=None, process=self.id, machine=None, event='Job completed')
