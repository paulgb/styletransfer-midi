#!/usr/bin/env python

from server import WEIGHTS_FILE, NUM_WEIGHTS


def main():
    with open(WEIGHTS_FILE) as f:
        for _ in range(NUM_WEIGHTS):
            print(ord(f.read(1)), end=' ')
    print()


if __name__ == '__main__':
    main()
