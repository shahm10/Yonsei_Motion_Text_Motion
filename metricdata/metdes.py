# import os
# import numpy as np

# folder_path = '/hdd1/undergraduate_research/cjw/LLMproject/metricData'

# npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# output_file_path = os.path.join(folder_path, 'metrics_output.txt')

# with open(output_file_path, 'w') as output_file:
#     # 각 npy 파일을 열고 각 행의 max, min, median, mean 계산하기
#     for npy_file in npy_files:
#         file_path = os.path.join(folder_path, npy_file)
#         data = np.load(file_path)  # .npy 파일 열기

#         # 파일 처리 정보 기록
#         output_file.write(f"Processing {npy_file}:\n")
        
#         # 각 행의 max, min, median, mean 계산
#         row_max = np.max(data, axis=0)
#         row_min = np.min(data, axis=0)
#         row_median = np.median(data, axis=0)
#         row_mean = np.mean(data, axis=0)
        
#         # 각 행의 결과를 텍스트 파일에 저장
#         for i in range(data.shape[1]):
#             output_file.write(f"Row {i}: max={row_max[i]}, min={row_min[i]}, median={row_median[i]}, mean={row_mean[i]}\n")
        
#         output_file.write("\n")  # 파일 간의 구분을 위한 빈 줄 추가


# print(f"Metrics saved to {output_file_path}")


metrics = {
    0: "Center of Mass distance",
    1: "Symmetry",
    2: "Grounding",
    3: "Arm fold",
    4: "Leg fold",
    5: "Kinetic Energy",
    6: "Potential Energy",
    7: "Bone length Coherence",
    8: "Torque",
    9: "Center Velocity",
    10: "Extremity speed",
    11: "Left arm extremity angular velocity",
    12: "Right arm extremity angular velocity",
    13: "Left leg extremity angular velocity",
    14: "Right leg extremity angular velocity",
    15: "Partial joint attention",
    16: "Efficiency",

}

import os
import numpy as np

folder_path = '/hdd1/undergraduate_research/cjw/LLMproject/metricData'
n = 10  # max min n% 설정 
m = 3  # m개의 연속된 indice 설정

def print_continuous_indices(indices, label, i):
            start_idx = 0
            while start_idx < len(indices):
                end_idx = start_idx
                # 연속된 인덱스를 확인
                while end_idx + 1 < len(indices) and indices[end_idx + 1] == indices[end_idx] + 1:
                    end_idx += 1
                
                # 연속된 인덱스가 m개 이상일 경우 출력
                if end_idx - start_idx + 1 >= m:
                    print(f"This is {metrics[i]} {label} peak: {indices[start_idx]} ~ {indices[end_idx]}")
                
                start_idx = end_idx + 1



npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

for npy_file in npy_files:
    file_path = os.path.join(folder_path, npy_file)
    data = np.load(file_path)  # .npy 파일 열기
    
    # 각 열에 대해 max n%, min n% 인덱스 계산
    for i in range(17): #data.shape[1]):
        column = data[:, i]
        sorted_indices = np.argsort(column)  # 값을 정렬하여 인덱스를 가져옴
        
        # 하위 n% 인덱스
        min_n_indices = sorted_indices[:int(len(column) * n / 100)]
        # 상위 n% 인덱스
        max_n_indices = sorted_indices[-int(len(column) * n / 100):]
        
        print(f"Column {i}:")
        print(f"  Min {n}% indices: {min_n_indices}, values: {column[min_n_indices]}")
        print(f"  Max {n}% indices: {max_n_indices}, values: {column[max_n_indices]}")
        # 상위 n%에 대해 연속된 m개 인덱스 출력
        print_continuous_indices(max_n_indices, 'Max',i)
        # 하위 n%에 대해 연속된 m개 인덱스 출력
        print_continuous_indices(min_n_indices, 'Min',i)
    
    print()  # 파일 간의 구분을 위한 빈 줄 출력



