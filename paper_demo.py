from locust import HttpUser, task

class BrutalDemoUser(HttpUser):
    # Notice: NO wait_time here. 
    # These users will fire requests as fast as your computer physically allows.

    @task
    def stress_test(self):
        # Targeting the heavy endpoint. 
        # (If your server expects data here, hitting it empty will still cause error-handling load)
        self.client.get("/predict")