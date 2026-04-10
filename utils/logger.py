from datetime import datetime
import csv
from typing import Dict, List
from .params import _LOGS_DIR


class LOGGER:
    INFO = "\033[0;37m[INFO]\033[0m "
    WARNING = "\033[0;33m[WARNING]\033[0m "
    ERROR = "\033[0;31m[ERROR]\033[0m "
    DEBUG = "\033[0;32m[DEBUG]\033[0m "


def csv_init(model_name: str, params: List):
    # Uncomment these lines if need timestamp for file name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    csv_filename = _LOGS_DIR / f"{model_name}_{timestamp}.csv"

    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(params)

    return csv_filename


def csv_logger(csv_filename, params: Dict):
    with open(csv_filename, "r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)

    with open(csv_filename, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writerow(params)


if __name__ == "__main__":
    # Simple test
    headers = ["time", "value_1", "value_2"]
    filename = csv_init("test_logger", headers)
    print(f"Log file created at: {filename}")

    for i in range(5):
        data = {"time": i * 0.1, "value_1": i, "value_2": i**2}
        csv_logger(filename, data)
        print(f"Logged row {i}: {data}")
