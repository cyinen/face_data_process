import os
import csv
import random
import stat

exp_times = 100000

gaussian_mean = 30
gaussian_variance = 4

QP_min = 22
QP_max = 38
QP_step = 2

static_dict = {}
for i in range(QP_min, QP_max + QP_step, QP_step):
    static_dict[str(i)] = 0

for i in range(exp_times):
    random_num = random.gauss(gaussian_mean, gaussian_variance)
    random_qp = round(random_num / QP_step) * QP_step
    while(random_qp < QP_min or random_qp > QP_max):
        random_num = random.gauss(gaussian_mean, gaussian_variance)
        random_qp = round(random_num / QP_step) * QP_step
    static_dict[str(random_qp)] += 1

csv_file_path = f"./gaussian_statistical_results_{gaussian_variance}_x{exp_times}.csv"
with open(csv_file_path,"w") as csv_file:
    writer = csv.DictWriter(csv_file, static_dict.keys())
    writer.writeheader()
    writer.writerow(static_dict)