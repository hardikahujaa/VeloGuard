from locust import HttpUser, task, between, LoadTestShape
import math
import random

class LoadGuardUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def access_api(self):
        if random.random() < 0.2:
            self.client.get(f"/predict?cb_nonce={random.randint(1, 1000000)}")
        else:
            self.client.get("/predict")

class UltimateChaosShape(LoadTestShape):
    time_limit = 1200  # 20 minutes

    def __init__(self):
        super().__init__()
        self.current_state = "normal"
        self.state_end_time = 0
        self.state_start_time = 0
        self.states = ["flash_crowd", "sniper", "ramp", "wave", "pulse", "stochastic"]

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        if run_time >= self.state_end_time:
            if self.current_state != "normal" and random.random() < 0.4:
                self.current_state = "normal"
            else:
                self.current_state = random.choice(self.states)

            self.state_end_time   = run_time + random.randint(30, 90)
            self.state_start_time = run_time

        time_in_state = run_time - self.state_start_time

        if self.current_state == "normal":
            return (30, 10)
        elif self.current_state == "flash_crowd":
            return (200, 50)
        elif self.current_state == "sniper":
            if time_in_state < 8:
                return (300, 300)
            else:
                self.state_end_time = run_time
                return (10, 10)
        elif self.current_state == "ramp":
            user_count = int(30 + (time_in_state * 2.5))
            return (user_count, 10)
        elif self.current_state == "wave":
            user_count = int(100 + (80 * math.sin(time_in_state / 5.0)))
            return (user_count, 50)
        elif self.current_state == "pulse":
            if int(time_in_state) % 10 < 3:
                return (250, 200)
            else:
                return (20, 50)
        elif self.current_state == "stochastic":
            return (random.randint(50, 250), 100)