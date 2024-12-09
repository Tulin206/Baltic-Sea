import yaml

params = {
    "learning_rate": 0.01,
    "batch_size": 16
}

with open("params_v1.yaml", "w") as file:
    yaml.dump(params, file)
