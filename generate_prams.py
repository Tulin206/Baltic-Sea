import yaml

params = {
    "learning_rate": 0.01,
    "batch_size": 16
}

with open("params.yaml", "w") as file:
    yaml.dump(params, file)
