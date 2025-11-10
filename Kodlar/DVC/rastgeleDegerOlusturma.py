# -*- coding: utf-8 -*-
"""
@author: SemihAcmali
"""

import csv
import random
from pathlib import Path


def generate_experience_salary_dataset(
    output_csv_path: Path,
    num_rows: int = 2000,
    seed: int = 13,
) -> None:
    """Generate a synthetic dataset of experience (years) and salary.

    - deneyim_yili: float in [0, 40], rounded to 2 decimals
    - maas: base + slope * deneyim_yili + noise, rounded to int and clipped to >= 10000
    """
    random.seed(seed)

    base_salary = 15000.0
    salary_per_year = 3500.0
    noise_std = 10000.0

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["deneyim_yili", "maas"])  # header

        for _ in range(num_rows):
            years = round(random.uniform(0.0, 40.0), 2)
            # Gaussian noise using Box-Muller via random.gauss if available
            noise = random.gauss(0.0, noise_std)
            salary = base_salary + salary_per_year * years + noise
            salary = max(10000.0, salary)
            writer.writerow([years, int(round(salary))])


if __name__ == "__main__":
    output_path = Path("deneyim_maas.csv")
    generate_experience_salary_dataset(output_path)
    print(f"Oluşturuldu: {output_path.resolve()} (2000 satır)")

