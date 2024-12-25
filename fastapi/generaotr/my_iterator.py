
class Countdown(object):
    def __init__(self, start):
        self.iter = CountdownIterator(start)
    def __iter__(self):
        return self.iter


class CountdownIterator(object):
    def __init__(self, start):
        self.start = start
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start

counter = Countdown(5)
for i in counter:
    print(i)
