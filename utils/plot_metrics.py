import os

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


def plot_mlflow_metrics(run_id: str):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(root_dir, "mlflow_logs")
    os.makedirs(save_dir, exist_ok=True)

    client = MlflowClient()
    data = client.get_run(run_id).data
    metric_keys = list(data.metrics.keys())

    if not metric_keys:
        print(f"[MLflow] Нет метрик для run_id={run_id}")
        return

    for metric in metric_keys:
        history = client.get_metric_history(run_id, metric)
        if not history:
            continue

        steps = [m.step for m in history]
        values = [m.value for m in history]

        plt.figure()
        plt.plot(steps, values, label=metric)
        plt.title(f"{metric} over time")
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.legend()

        output_path = os.path.join(save_dir, f"{metric}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"[MLflow] Сохранён график: {output_path}")
