#!/usr/bin/env python

import mmap
from select import select
from sys import stdin, stdout
from os import path

WEIGHTS_FILE = 'weights.dat'
RESULTS_FILE = 'results.json'
NUM_WEIGHTS = 16
TIMEOUT = 0.5


def main():
    if not path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, 'w+b') as f:
            f.write(b'\0' * NUM_WEIGHTS)

    if not path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w'):
            pass

    results_lastmodtime = None

    with open(WEIGHTS_FILE, 'a+b') as weights_f:
        weights_mm = mmap.mmap(weights_f.fileno(), 0)

        try:
            while True:
                r = select([stdin], [], [], TIMEOUT)
                if r[0]:

                    line = stdin.readline()
                    try:
                        index, value = line.strip().split(' ')
                        index = int(index)
                        value = int(value)
                        assert 0 <= index < 16
                        assert 0 <= value < 128
                        weights_mm[index] = value
                    except ValueError:
                        pass

                results_time = path.getmtime(RESULTS_FILE)
                if results_time != results_lastmodtime:
                    with open(RESULTS_FILE) as rf:
                        print(rf.read())
                    stdout.flush()
                    results_lastmodtime = results_time

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
