import time
from typing import Deque, Tuple
from collections import deque
import random
import unittest

from model_api import sleep_if_reach_RPM, sleep_if_reach_TPM


class TestSleep(unittest.TestCase):
    def setUp(self):
        self.tpm = 10
        self.rpm = 20
        self.tokens_of_prompts = [4, 2, 3, 1, 2, 2, 5, 1, 3, 4]
        self.times = [1.5, 2.5, 0.5, 2.7, 3.1, 1.8, 1.1, 1.7, 0.2, 1.2]

    def test_sleep_if_reach_TPM(self):
        print("test_sleep_if_reach_TPM")

        tokens_in_last_minute = []

        prompt_logs: Deque[Tuple[float, int]] = deque()

        for i, tokens in enumerate(self.tokens_of_prompts):
            sleep_if_reach_TPM(self.tpm, tokens, prompt_logs)

            t = time.time()
            prompt_logs.appendleft((t, tokens))
            tokens_in_last_minute.append(sum([t for _, t in prompt_logs]))

            time.sleep(self.times[i])

        expected = [4, 6, 9, 10, 8, 10, 10, 10, 9, 8]
        self.assertEqual(tokens_in_last_minute, expected)
        
        for tokens in tokens_in_last_minute:
            self.assertLessEqual(tokens, self.tpm)

    def test_sleep_if_reach_RPM(self):
        print("test_sleep_if_reach_RPM")

        timestamps = []

        timestamp = 0
        offset = -time.time()

        for i in range(10):
            timestamp = sleep_if_reach_RPM(self.rpm, timestamp)

            timestamps.append(timestamp + offset)
            time.sleep(self.times[i])

        for i in range(1, len(timestamps)):
            self.assertGreaterEqual(timestamps[i] - timestamps[i - 1], 60 / self.rpm)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
