import numpy as np
import json
import os
import re
import argparse

def extract_number_from_filename(filename):
    # Extract number from the filename using regex, if any
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def find_ranges_by_threshold(data, max_percentage=0.8, min_percentage=0.2):
    # Calculate absolute maximum and minimum
    abs_max = np.max(data)
    abs_min = np.min(data)

    # Set thresholds for maxima and minima
    max_threshold = max_percentage * abs_max  # Detect values near the absolute maximum
    min_threshold = abs_min + min_percentage * (abs_max - abs_min)  # Detect values near the absolute minimum

    # Find ranges where data is greater than or equal to the max threshold
    maxima_ranges = []
    minima_ranges = []

    # Helper function to identify ranges
    def get_ranges(indices):
        ranges = []
        start = indices[0]
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:  # If not consecutive
                ranges.append([start, indices[i - 1]])
                start = indices[i]
        ranges.append([start, indices[-1]])  # Add the last range
        return ranges

    # Identify indices where the data is above or below the thresholds
    maxima_indices = np.where(data >= max_threshold)[0].tolist()
    minima_indices = np.where(data <= min_threshold)[0].tolist()

    # Get ranges for maxima and minima
    if maxima_indices:
        maxima_ranges = get_ranges(maxima_indices)
    if minima_indices:
        minima_ranges = get_ranges(minima_indices)

    # Extract values for these ranges
    maxima_values = [float(np.max(data[start:end + 1])) for start, end in maxima_ranges]
    minima_values = [float(np.min(data[start:end + 1])) for start, end in minima_ranges]

    return maxima_ranges, maxima_values, minima_ranges, minima_values

def process_npy_file_for_com(file_path, instruction_template, metric_idx, max_percentage=0.8, min_percentage=0.2):
    # Load the .npy file
    data = np.load(file_path)

    # Print the file being processed
    print(f"Processing file: {os.path.basename(file_path)}")

    # Extract the data for the specified metric
    metric_data = np.round(data[:, metric_idx] * 100).astype(int)  # Scale by 100 and convert to integers

    # Find local maxima and minima based on threshold
    maxima_ranges, maxima_values, minima_ranges, minima_values = find_ranges_by_threshold(
        metric_data, max_percentage=max_percentage, min_percentage=min_percentage)

    # Total length of the frame
    total_length = len(metric_data)

    # Generate textual output
    maxima_text = f"Local maxima: {maxima_ranges}."
    minima_text = f"Local minima: {minima_ranges}."

    total_length_text = f" Frame length is: {total_length}"

    dynamic_instruction = instruction_template + total_length_text
    
    textual_output = (
        f"{maxima_text}."
        f"{minima_text}."
    )

    # Create the output dictionary in the required format
    output_dictionary = {
        "instruction": dynamic_instruction, 
        "integer": [int(x) for x in metric_data.tolist()],  # Convert to native Python int
        "output": textual_output  # Include the textual description
    }

    return output_dictionary


def generate_combined_output(folder_path, max_percentage=0.8, min_percentage=0.2):
    # Define instructions directly in the script
    instructions = {
        1:     "CoM is the Center of Mass. The closer to the max value, the more stable the posture, "
    "and the closer to the min value, the more unstable the posture. \n\n"
    "First, what is the length of the frame? "
    "Second, locate the frame's local maxima and specify their values, particularly where the changes are rapid. "
    "Third, locate the frame's local minima and specify their values, particularly where the changes are rapid. "
    "Lastly, what are the characteristics of the motion within those intervals?",
        2: "Symmetry represents the body is divided into left and right sides, and the similarity between the symmetrical joints is calculated. Near the maximum value, the less symmetry there is between the left and right sides of the body with moving only one side of arm or leg, and near the minimum value, the more symmetry there is between the left and right sides of the body.",
        3: "Grounding represents whether and to what extent both feet are grounded. Near the maximum value, both feet are in contact with the ground, and near the minimum value, both legs are far from the ground with a large extent equals to jump and small extent equals to walk. Specifically, a peak value of 0.7 corresponds to actions like jumping, 0.9 to  actions like walking or running.",
        4: "Arm fold represents a quantification of the angle of the arm (wrist-elbow-shoulder angle). Near the maximum value, both arms are fully extended, and near the minimum value, both arms are folded.",
        5: "Leg fold represents a numerical representation of the angle of the legs (ankle-knee-pelvis angle). Near the maximum value, the legs are fully extended, and near the minimum value, the legs are folded.",
        6: "Kinetic energy represents the kinetic energy of the whole body. Near the maximum value, the movement is dynamic, indicating that the body is actively performing an action. Near the minimum value, the movement is static, suggesting that the body is preparing to act.",
        7: "Potential energy represents the height level of the body's center of mass. Near the maximum value, it can be considered a jumping position, and near the minimum value, it can be considered a sitting position. Specifically, a value of 0.4 typically corresponds to the initial standing position. At a peak value of 1 corresponds to actions like significant jump, 0.7 to actions like small jump, 0 to actions like sitting.",
        8: "Bone length coherence represents how well the initial inter-articular length is maintained throughout the sequence. Near the maximum value, similar to the initial bone length, near the minimum value, different from the initial bone length. If bone length consistency is measured low, people may question the integrity of the data.",
        9: "Torque represents the torque value on the limbs (both arms, both legs). Near the maximum value, the torque value on the body is higher (more movement) indicating that the body is actively performing an action. Near the minimum value, the torque value on the body is lower (less movement).",
        10: "Center velocity represents the speed of movement of the center of gravity of the body. Near the maximum value, the faster the center of the body moves, the closer to the minimum value, the slower the center of the body moves. Specifically, a peak value of 1 corresponds to actions like jumping, 0.6 to jumps within specific actions like ballet jumps or cartwheels, 0.4 to sitting, and 0.2 to walking.",
        11: "Extremity speed represents the speed of movement of the end joints of the limbs (both wrists, ankles). Near the maximum value, the faster the speed of the extremities, near the minimum value, the less movement of the extremities.",
        12: "Left arm extremity angular velocity represents the angular velocity value of left arm. Near the maximum value, the larger the angular velocity value of left arm, indicating fast rotation, near the minimum value, the slower the rotation.",
        13: "Right arm extremity angular velocity represents the angular velocity value of right arm. Near the maximum value, the larger the angular velocity value of right arm, indicating fast rotation, near the minimum value, the slower the rotation.",
        14: "Left leg extremity angular velocity represents the angular velocity value of left leg. Near the maximum value, the larger the angular velocity value of left leg, indicating fast rotation, near the minimum value, the slower the rotation.",
        15: "Right leg extremity angular velocity represents the angular velocity value of right leg. Near the maximum value, the larger the angular velocity value of right leg, indicating fast rotation, near the minimum value, the slower the rotation.",
    }

    combined_results = []

    # Automatically process all .npy files in the specified folder
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
    
    # Sort files numerically based on extracted numbers from filenames
    file_list.sort(key=extract_number_from_filename)

    # Process each metric index with its corresponding instruction
    for metric_idx, (instruction_number, instruction_template) in enumerate(instructions.items()):
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)  # Files are in the specified folder path
            result = process_npy_file_for_com(
                file_path, 
                instruction_template, 
                metric_idx, 
                max_percentage=max_percentage, 
                min_percentage=min_percentage
            )
            combined_results.append(result)  # Append results for each file and metric

    # Save all results to a single JSON file
    output_file = f"combined_output.json"
    save_results_to_json(combined_results, output_file)
    print(f"Combined output saved to {output_file}")

def save_results_to_json(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, separators=(',', ': '))

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process .npy files to find local maxima and minima for different metrics and combine outputs.")
    parser.add_argument('--max_percentage', type=float, default=0.8, help='Percentage threshold for local maxima detection.')
    parser.add_argument('--min_percentage', type=float, default=0.2, help='Percentage threshold for local minima detection.')

    # Parse arguments
    args = parser.parse_args()

    # Define the folder path containing all .npy files
    folder_path = '/Users/SallyHome/Documents/AIStudies/Yonsei/metricdata'

    # Generate combined output with provided arguments
    generate_combined_output(
        folder_path=folder_path,
        max_percentage=args.max_percentage,
        min_percentage=args.min_percentage
    )

if __name__ == "__main__":
    main()

# def generate_dataset_for_com(metric_idx, instruction_number, folder_path, max_percentage=0.8, min_percentage=0.2):
#     # Define instructions directly in the script
#     instructions = {


#     # Validate instruction number
#     if instruction_number not in instructions:
#         print(f"Error: Instruction number {instruction_number} is not valid.")
#         return

#     instruction_template = instructions[instruction_number]

#     all_results = []
    
#     # Automatically process all .npy files in the specified folder
#     file_list = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
    
#     # Sort files numerically based on extracted numbers from filenames
#     file_list.sort(key=extract_number_from_filename)

#     for file_name in file_list:
#         file_path = os.path.join(folder_path, file_name)  # Files are in the specified folder path
#         result = process_npy_file_for_com(file_path, instruction_template, metric_idx, max_percentage=max_percentage, min_percentage=min_percentage)
#         all_results.append(result)  # Append results from each file

#     # Automatically create output file name based on instruction number
#     output_file = f"version2_output_metric_{metric_idx + 1}_instruction_{instruction_number}.json"
    
#     # Save results to JSON file
#     save_results_to_json(all_results, output_file)
#     print(f"Output saved to {output_file}")

# def save_results_to_json(results, output_file):
#     with open(output_file, 'w') as f:
#         json.dump(results, f, indent=4, separators=(',', ': '))

# def main():
#     # Set up argument parsing
#     parser = argparse.ArgumentParser(description="Process .npy files to find local maxima and minima for different metrics.")
#     parser.add_argument('--metric_idx', type=int, required=True, help='Index of the metric to be processed (0-based).')
#     parser.add_argument('--max_percentage', type=float, default=0.8, help='Percentage threshold for local maxima detection.')
#     parser.add_argument('--min_percentage', type=float, default=0.2, help='Percentage threshold for local minima detection.')
#     parser.add_argument('--instructions', type=int, required=True, help='Instruction number corresponding to the metric.')

#     # Parse arguments
#     args = parser.parse_args()

#     # Define the folder path containing all .npy files
#     folder_path = '/Users/SallyHome/Documents/AIStudies/Yonsei/metricdata'

#     # Generate dataset with provided arguments
#     generate_dataset_for_com(
#         metric_idx=args.metric_idx,
#         instruction_number=args.instructions,
#         folder_path=folder_path,
#         max_percentage=args.max_percentage,
#         min_percentage=args.min_percentage
#     )

# if __name__ == "__main__":
#     main()
