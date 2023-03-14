import multiprocessing

class DeepLearningModule(Process):
    def __init__(self, queue, result_q):
        Process.__init__(self)
        self.queue = queue
        self.result_q = result_q

    def run(self):
        while True:
            start_val = self.queue.get()
            try:
                # dummy code. Real code has network calls here
                thread_output = [ri(0, 10) + start_val, ri(0, 10) + start_val, ri(0, 10) + start_val]
                self.result_q.put(thread_output)
            finally:
                self.queue.task_done()