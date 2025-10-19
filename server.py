import flwr as fl

# Function to pass round number to clients during fit
def get_fit_config(server_round: int):
    return {"round": server_round}

# Function to pass round number to clients during evaluate
def get_eval_config(server_round: int):
    return {"round": server_round}

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=1,   # adjust to your setup
        min_fit_clients=1,
        min_evaluate_clients=1,
        on_fit_config_fn=get_fit_config,       # <--- added
        on_evaluate_config_fn=get_eval_config, # <--- added
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
