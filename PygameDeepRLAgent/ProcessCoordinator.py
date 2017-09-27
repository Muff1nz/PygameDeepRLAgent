#from multiprocessing import Lock

# Meant to work like tf.train.Coordinator() for processes
class ProcessCoordinator:
    def __init__(self):
        pass
        #self.lock = Lock()
        #self.shouldStop = False

    def should_stop(self):
        return False
        #with self.lock:
        #    return self.should_stop()

    def request_stop(self):
        pass
        #with self.lock:
        #    self.shouldStop = True

    def join(self, processes):
        for process in processes:
            process.join()
