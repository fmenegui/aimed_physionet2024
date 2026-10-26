import sys

class Tee(object):
    def __init__(self, name):
        self.file = open(name, 'w')
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)
        self.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()
