import argparse, os, datetime, gc, yaml
from ortools.linear_solver import pywraplp
import yaml
import numpy as np
import yaml
import numpy as np


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def write_yaml_file(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def get_mixed_precision_config_weight(stage='sqnr', ratio_config=None, sensitivity_config=None, mean_bit=None):

    sensitivity_config_tmp = {}
    for key, value in sensitivity_config.items():
        if stage == 'sqnr':
            sensitivity_config_tmp[key] = value
    
    sensitivity_config = sensitivity_config_tmp
    print(len(sensitivity_config))

    if stage == 'sqnr':
        b_values = [8]    # bit width candidate 

    # creat a solver
    mean_bit = float(mean_bit)
    solver = pywraplp.Solver.CreateSolver('SCIP')
    
    w = ratio_config # weight ratio dict
    s = sensitivity_config   # sensitivity dict
    c = {}

    # 计compute the para size
    import numpy as np
    intensity = 0
    intensity = sum(w[name] for name, sen in s.items())


    # create the variable
    for name, sen in s.items():
        for b in b_values:
            c[(name, b)] = solver.BoolVar('c_' + name + '_' + str(b))
    print("Number of variables =", solver.NumVariables())


    # create the constrains
    for name, sen in s.items():
        solver.Add(sum(c[(name, b)] for b in b_values) == 1)
    print("Number of constraints =", solver.NumConstraints())

    solver.Add(sum(sum(c[(i, b)] * b * w[i] for i,sen in s.items()) for b in b_values) >= (mean_bit - 0.02) * intensity)
    solver.Add(sum(sum(c[(i, b)] * b * w[i] for i,sen in s.items()) for b in b_values) <= (mean_bit + 0.02) * intensity)
    print("Number of constraints =", solver.NumConstraints())


    import math
    objective = solver.Objective()
    for name, sen in s.items():
        # print(name,ssim)
        for b in b_values:
            objective.SetCoefficient(c[(name, b)], sen[int(math.log2(b)-1)])  # s_{i,b}是c_{i,b}的系数
    objective.SetMaximization()


    # solve the problem
    solver.Solve()

    # generate the bit width config for weight and act seperatly
    print('Solution:')
    solution_dict = {}
    for name, sen in s.items():
        solution_dict[name] = {}
        solution_dict[name] = 0
        solution_dict[name] = 0
        for b in b_values:
            if c[(name, b)].solution_value() > 0:
                solution_dict[name] = b

    return solution_dict


# def get_mixed_precision_config_act(stage, ratio_config, sensitivity_config, mean_bit, act_sensitivity_1):

#     layer_filtered_ratio = {}
#     for key, value in ratio_config.items():
#         if key in act_sensitivity_1:
#             layer_filtered_ratio[key] = value


#     layer_filtered_para = sum(para for name, para in layer_filtered_ratio.items())
#     print(layer_filtered_para)


#     sensitivity_config_tmp = {}
#     for key, value in sensitivity_config.items():
#         if stage == 'sqnr':
#             sensitivity_config_tmp[key] = value


#     sensitivity_config = sensitivity_config_tmp
#     print(len(sensitivity_config))

#     # create a solver
#     mean_bit = float(mean_bit)
#     solver = pywraplp.Solver.CreateSolver('SCIP')

#     # define the variable
#     w = ratio_config # weight ratio dict
#     s = sensitivity_config   # sensitivity dict
#     b_values = [4, 8]    # bit width candidate 
#     c = {}

#     intensity = sum(w[name] for name, ssim in s.items())
#     print(intensity)


#     for name, ssim in s.items():
#         for b in b_values:
#             c[(name, b)] = solver.BoolVar('c_' + name + '_' + str(b))
#     print("Number of variables =", solver.NumVariables())


#     for name, ssim in s.items():
#         solver.Add(sum(c[(name, b)] for b in b_values) == 1)
#     print("Number of constraints =", solver.NumConstraints())

#     solver.Add(sum(sum(c[(i, b)] * b * w[i] for i,ssim in s.items()) for b in b_values) >= (mean_bit - 0.02) * intensity)
#     solver.Add(sum(sum(c[(i, b)] * b * w[i] for i,ssim in s.items()) for b in b_values) <= (mean_bit + 0.02) * intensity)
#     print("Number of constraints =", solver.NumConstraints())

#     import math
#     objective = solver.Objective()
#     for name, ssim in s.items():
#         # print(name,ssim)
#         for b in b_values:
#             objective.SetCoefficient(c[(name, b)], ssim[int(math.log2(b)-1)]) 
#     objective.SetMaximization()


#     solver.Solve()

#     # generate the bit width config for weight and act seperatly
#     print('Solution:')
#     solution_dict = {}
#     for name, ssim in s.items():
#         solution_dict[name] = {}
#         solution_dict[name] = 0
#         solution_dict[name] = 0
#         for b in b_values:
#             if c[(name, b)].solution_value() > 0:
#                 solution_dict[name] = b

#     return solution_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixed_precision_type", required=True,
        type=str,
        help="perform integer programming for weight or act"
    )
    parser.add_argument(
        "--sensitivity", required=True,
        type=str,
        help="path for sensitivity list based on sqnr"
    )
    parser.add_argument(
        "--para_size_config", required=True,
        type=str,
        help="path for the config of parameter size"
    )
    parser.add_argument(
        "--mixed_precision_config", required=True,
        type=str,
        help="path for the output config"
    )
    parser.add_argument(
        "--target_bitwidth", required=True,
        type=float,
        help="the average bitwidth of weight or act"
    )
    opt = parser.parse_args()


    if opt.mixed_precision_type == 'weight':

        os.makedirs(opt.mixed_precision_config, exist_ok=True)

        # config_path_weight_ssim ='../sensitivity_log/sdxl_turbo/weight/ssim/bs32_split_ssim_weight/sensitivity.yaml'
        with open(opt.sensitivity, 'r') as file:
            layer_config = yaml.safe_load(file)

        # config_path = f'./tensor_ratio/sdxl_turbo/weight_ratio_config.yaml'
        with open(opt.para_size_config, 'r') as file:
            ratio_config = yaml.safe_load(file)

        bitwidth_list = np.linspace(opt.target_bitwidth-0.3, opt.target_bitwidth, 10)

        for average_bitwidth in bitwidth_list:
            # for k in layer_ratios:
            
            weight_config = get_mixed_precision_config_weight(stage='sqnr', ratio_config=ratio_config, sensitivity_config=layer_config, mean_bit=average_bitwidth)
            bit_value = format(average_bitwidth,'.2f')

            write_yaml_file(file_path=os.path.join(opt.mixed_precision_config, f"weight_{(bit_value)}.yaml"), data=weight_config)


