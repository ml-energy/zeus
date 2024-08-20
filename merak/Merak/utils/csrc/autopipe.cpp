// Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
//
// Maintainer: Wjliu (mcmillantac@163.com)
// Algorithm of paper: < AutoPipe: A Fast Pipeline Parallelism Approach
// with Balanced Partitioning and Micro-batch Slicing >
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Algorithm for auto pipeline partition according to critical path for sychronized pipeline.
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <map>
#include <time.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;

// Constants.
/***************************Begin******************************/
// Communication overhead for a vector with size [b, s, h](us).
const long long kCommunicationOverhead = 0;
/*************************** End ******************************/

// Return detail information about block partition for kNumPipelineStages.
vector<vector<int>> BlockPartitionAlgorithm(vector<int>& model, int& num_pipeline_stages, vector<vector<long long>>& block_time_mapping);

// Reconstruct the dynamic programming in a recursive way and put the result in block_partition variable.This process can be finish by the following two functions.The ReconstructBlockPartitionsFrontToBack function searches the result from front to back during reconstruct, which has a trend to put the stage of critical path later, while the ReconstructBlockPartitionsBackToFront function searches the result from back to front during reconstruct, which has a trend to put the stage of critical path earlier.
void ReconstructBlockPartitionsFrontToBack(vector<int>& model, vector<long long>& prefix_sum, vector<vector<long long>>& dp, int remaining_model_blocks, int remaining_partitions, vector<vector<int>>& block_partition);

void ReconstructBlockPartitionsBackToFront(vector<int>& model, vector<long long>& prefix_sum, vector<vector<long long>>& dp, int remaining_model_blocks, int remaining_partitions, vector<vector<int>>& block_partition);

// Return time cost and corresponding stage of critical path when using the block partition during training.
pair<long long, int> TrainingTimeCalculation(vector<vector<int>>& block_partition, vector<vector<long long>>& block_time_mapping);

// Put the forward time, backward time and last microbatch index for each stage of current block partition into corresponding array.
void CalculationForRelevantArray(vector<vector<int>>& block_partition, vector<vector<long long>>& block_time_mapping, vector<long long>& forward_time, vector<long long>& backward_time, vector<int>& last_microbatch);


// Calculate the start time of every forward/backward in 1F1B steady phase.Return the start time of the last forward microbatch on the critical path's steady phase and the corresponding stage.
pair<long long, int> CalculationForSteadyPhase(vector<int>& last_batch,vector<long long>& forward_time, vector<long long>& backward_time);

// Calculate the start time of last backward for stage of critical path and return the start time.
long long CalculationForCooldownPhase(int& num_pipeline_stages, int& stage_of_critical_path, long long& last_forward_microbatch_start_time, vector<long long>& forward_time, vector<long long>& backward_time);

// Output the block partition and relative results on the screen.
void OutputResult(vector<vector<int>>& block_partition, int& stage_of_critical_path, long long& block_partition_cost);

// Find minimum time cost overall block partitions for kNumPipelineStages with heuristic algorithm.Return the optimal block partition and corresponding time costs.
vector<vector<int>> FindBestBlockPartition(vector<vector<long long>>& block_time_mapping, int& num_pipeline_stages, vector<vector<int>>& initial_block_partition, int& stage_of_critical_path, long long& minimum_time_costs, vector<long long>& prefix_sum, vector<vector<long long>>& dynamic_programming_array);

// Get the prefix sum array and corresponding dynamic programming array of the model.
void GetPrefixSumArrayAndDpArray(vector<int>& model, int& num_pipeline_stages, vector<vector<long long>>& block_time_mapping, vector<long long>& prefix_sum, vector<vector<long long>>& dynamic_programming_array);

//The interface function, input the model's array of computation time or array of computation volumes, and the number of pipeline stages to be divided, the function returns the optimal pipeline partition.
vector<int> MerakPipe(vector<long long> model_calculation_forward, vector<long long> model_calculation_backward, int num_pipeline_stages);

// int main(){
//     vector<long long> model_cal_f = {1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,};
//     vector<long long> model_cal_b = {1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,};
//     int pp = 4;
//     vector<int> result = MerakPipe(model_cal_f, model_cal_b, pp);
//     return 0;
// }

vector<int> MerakPipe(vector<long long> model_calculation_forward, vector<long long> model_calculation_backward, int num_pipeline_stages){
    // block computing time mapping for different layers.For the first axis,
    vector<vector<long long>> block_time_mapping = {model_calculation_forward, model_calculation_backward};
    // transformer model architecture array.
    vector<int> model(model_calculation_forward.size());
    // initial the architecture array.
    for(int i = 0; i < model.size(); i++) {
        model[i] = i;
    }
    // Do block partition for kNumPipelineStages.
    vector<vector<int>> block_partition = BlockPartitionAlgorithm(model, num_pipeline_stages, block_time_mapping);
    // Variables used by find best block partition function.
    vector<long long> prefix_sum;
    vector<vector<long long>> dynamic_programming_array;
    // Initilize the variables by invoking the function.
    GetPrefixSumArrayAndDpArray(model, num_pipeline_stages, block_time_mapping, prefix_sum, dynamic_programming_array);
    // Get the optimal block partition and corresponding time costs.
    long long minimum_time_costs;
    int stage_of_critical_path;
    // Time calculation variables.
    // clock_t start, end;
    // start = clock();
    vector<vector<int>> optimal_block_partition = FindBestBlockPartition(block_time_mapping, num_pipeline_stages, block_partition, stage_of_critical_path, minimum_time_costs, prefix_sum, dynamic_programming_array);
    // end = clock();
    // cout << "------------------------------------------------------------" << endl;
    // cout << "          Heuristic Optimal Model Block Partition           " << endl;
    // cout << "------------------------------------------------------------" << endl;
    // cout << "                Algorithm Time Cost:" << (double)(end - start) / CLOCKS_PER_SEC << " s " << endl;
    // cout << "------------------------------------------------------------" << endl;
    // OutputResult(optimal_block_partition, stage_of_critical_path, minimum_time_costs);
    vector<int> ret;
    for(auto& partition: optimal_block_partition) {
        ret.push_back(partition[0]);
    }
    return ret;
}

vector<vector<int>> FindBestBlockPartition(vector<vector<long long>>& block_time_mapping, int& num_pipeline_stages, vector<vector<int>>& initial_block_partition, int& stage_of_critical_path, long long& minimum_time_costs, vector<long long>& prefix_sum, vector<vector<long long>>& dynamic_programming_array) {
    // Initialize variables for algorithm.
    vector<int> last_microbatch(num_pipeline_stages, 0);
    vector<long long> forward_time(num_pipeline_stages + 2, 0), backward_time(num_pipeline_stages + 2, 0);
    // Variables to record the optimal partition during the whole algorithm.
    int stage_of_critical_path_for_optimal_block_partition = INT32_MAX;
    vector<vector<int>> optimal_block_partition;
    long long optimal_block_partition_time_cost = __LONG_LONG_MAX__;
    // Start algorithm.
    // A red black tree to record the results that has been calculated.
    map<vector<vector<int>>, int> record;
    // A queue to store block partitions that may be optimal.
    queue<vector<vector<int>>> possible_block_partitions;
    possible_block_partitions.push(initial_block_partition);
    while(!possible_block_partitions.empty()) {
        // Get the front element of queue.
        vector<vector<int>> cur_block_partition = possible_block_partitions.front();
        possible_block_partitions.pop();
        // Initilize the relavant variables corresbonding to cur_block_partition by invoking the function.
        CalculationForRelevantArray(cur_block_partition, block_time_mapping, forward_time, backward_time, last_microbatch);
        // Get the base time cost and stage of critical path for current block partition,init the relevant variables.
        pair<long long, int> cur_block_partition_time_cost_and_stage_of_critical_path_base = TrainingTimeCalculation(cur_block_partition, block_time_mapping);
        long long cur_block_partition_time_cost_base = cur_block_partition_time_cost_and_stage_of_critical_path_base.first;
        int cur_stage_of_critical_path_base = cur_block_partition_time_cost_and_stage_of_critical_path_base.second;

        if(optimal_block_partition_time_cost > cur_block_partition_time_cost_base) {
            optimal_block_partition = cur_block_partition;
            optimal_block_partition_time_cost = cur_block_partition_time_cost_base;
            stage_of_critical_path_for_optimal_block_partition = cur_stage_of_critical_path_base;
        }

        // Adjust the block partition after the current block partition's stage of critical path.
        vector<vector<int>> adjust_block_partition = cur_block_partition;
        int adjust_stage = cur_stage_of_critical_path_base + 1;
        bool continue_adjust = true;
        long long time_cost_for_adjust_stage, time_cost_for_stage_of_critical_path, gap = 0;
        while(continue_adjust) {
            if(adjust_stage == adjust_block_partition.size()) {
                CalculationForRelevantArray(adjust_block_partition, block_time_mapping, forward_time, backward_time, last_microbatch);
                pair<long long, int> cur_block_partition_time_cost_and_stage_of_critical_path_adjust = TrainingTimeCalculation(adjust_block_partition, block_time_mapping);
                long long cur_block_partition_time_cost_adjust = cur_block_partition_time_cost_and_stage_of_critical_path_adjust.first;
                int cur_stage_of_critical_path_adjust = cur_block_partition_time_cost_and_stage_of_critical_path_adjust.second;
                // OutputResult(adjust_block_partition, cur_stage_of_critical_path_adjust, cur_block_partition_time_cost_adjust);
                if(optimal_block_partition_time_cost > cur_block_partition_time_cost_adjust) {
                    optimal_block_partition = adjust_block_partition;
                    optimal_block_partition_time_cost = cur_block_partition_time_cost_adjust;
                    stage_of_critical_path_for_optimal_block_partition = cur_stage_of_critical_path_adjust;
                }
                continue_adjust = false;
            }else{
                CalculationForRelevantArray(adjust_block_partition, block_time_mapping, forward_time, backward_time, last_microbatch);
                pair<long long, int> cur_block_partition_time_cost_and_stage_of_critical_path_adjust = TrainingTimeCalculation(adjust_block_partition, block_time_mapping);
                long long cur_block_partition_time_cost_adjust = cur_block_partition_time_cost_and_stage_of_critical_path_adjust.first;
                int cur_stage_of_critical_path_adjust = cur_block_partition_time_cost_and_stage_of_critical_path_adjust.second;
                // OutputResult(adjust_block_partition, cur_stage_of_critical_path_adjust, cur_block_partition_time_cost_adjust);
                if(cur_stage_of_critical_path_adjust == cur_stage_of_critical_path_base) {
                    if(optimal_block_partition_time_cost > cur_block_partition_time_cost_adjust) {
                        optimal_block_partition = adjust_block_partition;
                        optimal_block_partition_time_cost = cur_block_partition_time_cost_adjust;
                        stage_of_critical_path_for_optimal_block_partition = cur_stage_of_critical_path_adjust;
                    }
                    // Adjust block partition for current adjust stage.
                    time_cost_for_adjust_stage = 0;
                    for(int stage = cur_stage_of_critical_path_adjust + 1; stage <= adjust_stage; stage++) {
                        time_cost_for_adjust_stage += forward_time[stage + 1] + backward_time[stage + 1];
                    }
                    time_cost_for_stage_of_critical_path = (adjust_stage - cur_stage_of_critical_path_adjust) * backward_time[cur_stage_of_critical_path_adjust + 1] + gap;
                    if(time_cost_for_adjust_stage <= time_cost_for_stage_of_critical_path) {
                        adjust_stage++;
                    }else{
                        // Do backtrack.Find the first pipeline stage to which blocks can be added from back to front.
                        int backtracking_stage;
                        // Do backtrack.Find the first pipeline stage to which blocks can be added from front to back.
                        for(backtracking_stage = cur_stage_of_critical_path_adjust + 1; backtracking_stage < adjust_stage; backtracking_stage++) {
                            long long forward_backward_time_cost_for_backtracing_stage = forward_time[backtracking_stage + 1] + backward_time[backtracking_stage + 1];
                            long long forward_backward_time_cost_for_stage_of_critical_path = forward_time[cur_stage_of_critical_path_adjust + 1] + backward_time[cur_stage_of_critical_path_adjust + 1];
                            // Take communication time into consideration.
                            if(backtracking_stage != adjust_block_partition.size() - 1) {
                                forward_backward_time_cost_for_backtracing_stage += kCommunicationOverhead;
                            }
                            if(cur_stage_of_critical_path_adjust != 0) {
                                forward_backward_time_cost_for_stage_of_critical_path += kCommunicationOverhead;
                            }
                            // Judgement for block movement.
                            if(forward_backward_time_cost_for_stage_of_critical_path > forward_backward_time_cost_for_backtracing_stage + block_time_mapping[0][adjust_block_partition[backtracking_stage + 1][0]] + block_time_mapping[1][adjust_block_partition[backtracking_stage + 1][0]]) {
                                // Make block movement.Also, remember to reset the adjust stage and increse the corresponding value to gap.
                                adjust_block_partition[backtracking_stage].push_back(adjust_block_partition[backtracking_stage + 1][0]);
                                adjust_block_partition[backtracking_stage + 1].erase(adjust_block_partition[backtracking_stage + 1].begin());
                                long long time_cost_for_baktracing_stage = 0;
                                for(int stage = cur_stage_of_critical_path_adjust + 1; stage <= backtracking_stage; stage++) {
                                    time_cost_for_baktracing_stage += forward_time[stage + 1] + backward_time[stage + 1];
                                }
                                time_cost_for_stage_of_critical_path = (backtracking_stage - cur_stage_of_critical_path_adjust) * backward_time[cur_stage_of_critical_path_adjust + 1] + gap;
                                gap += max(0ll, time_cost_for_baktracing_stage - time_cost_for_stage_of_critical_path);
                                adjust_stage = backtracking_stage + 1;
                                break;
                            }
                        }
                        // If we do not find a stage to add one model block,ending search process.
                        if(backtracking_stage == adjust_stage) {
                            continue_adjust = false;
                        }
                    }
                }else{
                    continue_adjust = false;
                }
            }
        }

        // Try to move the stage of critical path to front stage by moving one block from current block partition's stage of critical path unless current stage of critical path is the first stage.
        if(cur_stage_of_critical_path_base > 0) {
            int from_stage = cur_stage_of_critical_path_base;
            // Get model blocks before the stage of critical path.
            vector<int> model_before_stage_of_critical_path;
            for(int stage = 0; stage < from_stage; stage++) {
                for(int& block: cur_block_partition[stage]) {
                    model_before_stage_of_critical_path.push_back(block);
                }
            }

            // Get the block partition that moves one model block forward.
            // Get the relative balance partition from dynamic programming array for block partition adjustion.
            model_before_stage_of_critical_path.push_back(cur_block_partition[from_stage][0]);
            vector<vector<int>> move_to_front;
            ReconstructBlockPartitionsFrontToBack(model_before_stage_of_critical_path, prefix_sum, dynamic_programming_array, model_before_stage_of_critical_path.size(), from_stage, move_to_front);
            // Reverse the order of model block partition as it is in the reverse order when reconstruct from the dynamic programming algorithm.
            reverse(move_to_front.begin(), move_to_front.end());
            // Pop the last element of model blocks.
            model_before_stage_of_critical_path.pop_back();

            // Complete model block partition.
            for(int stage = from_stage; stage < cur_block_partition.size(); stage++) {
                move_to_front.push_back(cur_block_partition[stage]);
            }
            // Remove the redundant model block.
            move_to_front[from_stage].erase(move_to_front[from_stage].begin());
            // The stage of critical path must move to front when we move a block to front,so we do not need to validate the stage of critical path after this operation.
            // Note that we remove redundancy during this process.
            if(!record.count(move_to_front)){
                possible_block_partitions.push(move_to_front);
                record[move_to_front] = 1;
            }

            // Get the block partition that moves one model block backward.
            // We can move one block to back only if the current stage of critical path is not the last stage.
            if(from_stage < cur_block_partition.size() - 1) {
                for(int block_index = 0; block_index < cur_block_partition[from_stage].size() - 1; block_index++) {
                    model_before_stage_of_critical_path.push_back(cur_block_partition[from_stage][block_index]);
                }
                // Get the relative balance partition from dynamic programming array for block partition adjustion.
                vector<vector<int>> move_to_back;
                ReconstructBlockPartitionsFrontToBack(model_before_stage_of_critical_path, prefix_sum, dynamic_programming_array, model_before_stage_of_critical_path.size(), from_stage + 1, move_to_back);
                // Reverse the order of model block partition as it is in the reverse order when reconstruct from the dynamic programming algorithm.
                reverse(move_to_back.begin(), move_to_back.end());
                // Complete model block partition.
                for(int stage = from_stage + 1; stage < cur_block_partition.size(); stage++) {
                    move_to_back.push_back(cur_block_partition[stage]);
                }
                // Add the block that is ignored during the preceding phase into correct partition.
                move_to_back[from_stage + 1].insert(move_to_back[from_stage + 1].begin(), cur_block_partition[from_stage].back());
                // Validate if the stage of critical path is less equal than original.Take this block partition into consideration only if its stage of critical path is less equal than original.
                pair<long long, int> move_to_back_block_partition_time_cost_and_stage_of_critical_path = TrainingTimeCalculation(move_to_back, block_time_mapping);
                if(move_to_back_block_partition_time_cost_and_stage_of_critical_path.second <= cur_stage_of_critical_path_base) {
                    // Note we need to remove the redundancy during this process.
                    if(!record.count(move_to_back)){
                        possible_block_partitions.push(move_to_back);
                        record[move_to_back] = 1;
                    }
                }
            }
        }
    }
    minimum_time_costs = optimal_block_partition_time_cost;
    stage_of_critical_path = stage_of_critical_path_for_optimal_block_partition;
    return optimal_block_partition;
}


void GetPrefixSumArrayAndDpArray(vector<int>& model, int& num_pipeline_stages, vector<vector<long long>>& block_time_mapping, vector<long long>& prefix_sum, vector<vector<long long>>& dynamic_programming_array) {
    int num_model_blocks = model.size();
    // Calculation for prefix sum array.
    prefix_sum.push_back(0);
    for(int& model_block: model) {
        prefix_sum.push_back(prefix_sum.back() + block_time_mapping[0][model_block] + block_time_mapping[1][model_block]);
    }
    // Vector for dynamic programming.
    vector<vector<long long>> dp(num_model_blocks + 1, vector<long long>(min(num_model_blocks, num_pipeline_stages) + 1, __LONG_LONG_MAX__));
    // Initial dp vector.
    dp[0][0] = 0;
    // Dynamic programming for num_pipeline_stages.
    for(int model_blocks = 1; model_blocks <= num_model_blocks; model_blocks++) {
        for(int partitions = 1; partitions <= min(model_blocks, num_pipeline_stages); partitions++) {
            for(int remaining_model_blocks = 0; remaining_model_blocks < model_blocks; remaining_model_blocks++) {
                // Note:Add startup overhead during partition.
                dp[model_blocks][partitions] = min(dp[model_blocks][partitions], max(dp[remaining_model_blocks][partitions - 1], prefix_sum[model_blocks] - prefix_sum[remaining_model_blocks]));
            }
        }
    }
    dynamic_programming_array = dp;
    return;
}

vector<vector<int>> BlockPartitionAlgorithm(vector<int>& model, int& num_pipeline_stages, vector<vector<long long>>& block_time_mapping) {
    vector<vector<int>> block_partition;
    int num_model_blocks = model.size();
    // Calculate the prefix sum for model blocks.There has three ways for computing prefix sum,forward time, backward time and forward&backward time.We use forward&backward time default.Noting there does not include forward/backward startup overhead,only considering computing time costs.
    vector<long long> prefix_sum;
    prefix_sum.push_back(0);
    for(int& model_block: model) {
        prefix_sum.push_back(prefix_sum.back() + block_time_mapping[0][model_block] + block_time_mapping[1][model_block]);
    }
    // Vector for dynamic programming.
    vector<vector<long long>> dp(num_model_blocks + 1, vector<long long>(min(num_model_blocks, num_pipeline_stages) + 1, __LONG_LONG_MAX__));
    // Initial dp vector.
    dp[0][0] = 0;
    // Dynamic programming for num_pipeline_stages.
    for(int model_blocks = 1; model_blocks <= num_model_blocks; model_blocks++) {
        for(int partitions = 1; partitions <= min(model_blocks, num_pipeline_stages); partitions++) {
            for(int remaining_model_blocks = 0; remaining_model_blocks < model_blocks; remaining_model_blocks++) {
                // Note:Add startup overhead during partition.
                dp[model_blocks][partitions] = min(dp[model_blocks][partitions], max(dp[remaining_model_blocks][partitions - 1], prefix_sum[model_blocks] - prefix_sum[remaining_model_blocks]));
            }
        }
    }
    // Reconstruct partition results for num_pipeline_stages.
    ReconstructBlockPartitionsFrontToBack(model, prefix_sum, dp, num_model_blocks, num_pipeline_stages, block_partition);
    // Reverse the order of block partition as it is reversed during recursive find.
    reverse(block_partition.begin(), block_partition.end());
    return block_partition;
}

void ReconstructBlockPartitionsBackToFront(vector<int>& model, vector<long long>& prefix_sum, vector<vector<long long>>& dp, int remaining_model_blocks, int remaining_partitions, vector<vector<int>>& block_partition) {
    if(remaining_model_blocks == 0 && remaining_partitions == 0) return;
    // Processing errors.
    if(remaining_model_blocks <= 0 || remaining_partitions <= 0 || remaining_model_blocks < remaining_partitions) {
        cout << "Error occur during reconstruct the partition result! Algorithm shutdown." << endl;
        return;
    }
    // Initialize the prev block end index for remainning blocks in dynamic programming procedure.
    int prev_block_end_index = remaining_model_blocks - 1;
    // Find the end index of prev block.
    while(prev_block_end_index >= 0 && (dp[remaining_model_blocks][remaining_partitions] != max(dp[prev_block_end_index][remaining_partitions - 1], prefix_sum[remaining_model_blocks] - prefix_sum[prev_block_end_index]))) {
        prev_block_end_index--;
    }
    // Reconstruct the result of last partition blocks in remainning blocks.
    vector<int> blocks_in_this_partition;
    for(int i = prev_block_end_index + 1; i <= remaining_model_blocks; i++) {
        blocks_in_this_partition.push_back(model[i - 1]);
    }
    block_partition.push_back(blocks_in_this_partition);
    // Recursive to find the remaining blocks' partitions.
    ReconstructBlockPartitionsBackToFront(model, prefix_sum, dp, prev_block_end_index, remaining_partitions - 1, block_partition);
    return;
}

void ReconstructBlockPartitionsFrontToBack(vector<int>& model, vector<long long>& prefix_sum, vector<vector<long long>>& dp, int remaining_model_blocks, int remaining_partitions, vector<vector<int>>& block_partition) {
    if(remaining_model_blocks == 0 && remaining_partitions == 0) return;
    // Processing errors.
    if(remaining_model_blocks <= 0 || remaining_partitions <= 0 || remaining_model_blocks < remaining_partitions) {
        cout << "Error occur during reconstruct the partition result! Algorithm shutdown." << endl;
        return;
    }
    // Initialize the prev block end index for remainning blocks in dynamic programming procedure.
    int prev_block_end_index = 0;
    // Find the end index of prev block.
    while(prev_block_end_index < remaining_model_blocks && (dp[remaining_model_blocks][remaining_partitions] != max(dp[prev_block_end_index][remaining_partitions - 1], prefix_sum[remaining_model_blocks] - prefix_sum[prev_block_end_index]))) {
        prev_block_end_index++;
    }
    // Reconstruct the result of last partition blocks in remainning blocks.
    vector<int> blocks_in_this_partition;
    for(int i = prev_block_end_index + 1; i <= remaining_model_blocks; i++) {
        blocks_in_this_partition.push_back(model[i - 1]);
    }
    block_partition.push_back(blocks_in_this_partition);
    // Recursive to find the remaining blocks' partitions.
    ReconstructBlockPartitionsFrontToBack(model, prefix_sum, dp, prev_block_end_index, remaining_partitions - 1, block_partition);
    return;
}


pair<long long, int> TrainingTimeCalculation(vector<vector<int>>& block_partition, vector<vector<long long>>& block_time_mapping) {
    int num_pipeline_stages = block_partition.size();
    int num_microbatches = num_pipeline_stages * 2;
    // Build last microbatch array for every pipeline stage.
    vector<int> last_microbatch(num_pipeline_stages);
    for(int i = 0; i < num_pipeline_stages; i++) {
        last_microbatch[i] = num_microbatches - num_pipeline_stages + i;
    }
    // Build forward time array and backward time array.
    vector<long long> forward_time(num_pipeline_stages + 2, 0), backward_time(num_pipeline_stages + 2, 0);
    for(int i = 1; i <= num_pipeline_stages; i++) {
        long long forward_time_for_stage_i = 0, backward_time_for_stage_i = 0;
        for(int& block_type:block_partition[i - 1]) {
            forward_time_for_stage_i += block_time_mapping[0][block_type];
            backward_time_for_stage_i += block_time_mapping[1][block_type];
        }
        forward_time[i] = forward_time_for_stage_i;
        backward_time[i] = backward_time_for_stage_i;
    }
    // Calculation for the 1F1B steady phase.The function will return the start time of the last forward microbatch on the critical path's steady phase and the corresponding stage.
    pair<long long, int> last_forward_microbatch_start_time_and_stage_of_critical_path = CalculationForSteadyPhase(last_microbatch, forward_time, backward_time);
    long long last_forward_microbatch_start_time = last_forward_microbatch_start_time_and_stage_of_critical_path.first;
    int stage_of_critical_path = last_forward_microbatch_start_time_and_stage_of_critical_path.second;
    // Processing error.
    if(last_forward_microbatch_start_time_and_stage_of_critical_path.first == __LONG_LONG_MAX__){
        cout << "Function CalculationForSteadyPhase return __LONG_LONG_MAX__ when calculate block partition:" << endl;
        return make_pair(__LONG_LONG_MAX__, -1);
    }
    // Calculation for cooldown phase.Trying to get the start time of last microbatch for stage of cirtical path.
    long long last_backward_microbatch_start_time_for_stage_of_critical_path = CalculationForCooldownPhase(num_pipeline_stages, stage_of_critical_path, last_forward_microbatch_start_time, forward_time, backward_time);
    // Calculation for pipeline flush time.Only need to calculate the backward time from stage of critical path to the first stage.
    long long pipeline_flush_time = last_backward_microbatch_start_time_for_stage_of_critical_path;
    for(int stage = stage_of_critical_path; stage > 0; stage--) {
        pipeline_flush_time += backward_time[stage + 1];
        // Take comminication overhead into consideration.
        pipeline_flush_time += kCommunicationOverhead;
    }
    // Add last backward time of first stage,as it does not send backward to previous stage,we do not add a communication overhead.
    pipeline_flush_time += backward_time[1];
    return make_pair(pipeline_flush_time, stage_of_critical_path);
}

void CalculationForRelevantArray(vector<vector<int>>& block_partition, vector<vector<long long>>& block_time_mapping, vector<long long>& forward_time, vector<long long>& backward_time, vector<int>& last_microbatch) {
    int num_pipeline_stages = block_partition.size();
    int num_microbatches = num_pipeline_stages * 2;
    // Build last microbatch array for every pipeline stage.
    for(int i = 0; i < num_pipeline_stages; i++) {
        last_microbatch[i] = num_microbatches - num_pipeline_stages + i;
    }
    // Build forward time array and backward time array.
    for(int i = 1; i <= num_pipeline_stages; i++) {
        long long forward_time_for_stage_i = 0, backward_time_for_stage_i = 0;
        for(int& block_type:block_partition[i - 1]) {
            forward_time_for_stage_i += block_time_mapping[0][block_type];
            backward_time_for_stage_i += block_time_mapping[1][block_type];
        }
        forward_time[i] = forward_time_for_stage_i;
        backward_time[i] = backward_time_for_stage_i;
    }
    return;
}


pair<long long, int>  CalculationForSteadyPhase(vector<int>& last_batch,vector<long long>& forward_time, vector<long long>& backward_time) {
    int num_pipeline_stages = last_batch.size();
    int num_microbatches = num_pipeline_stages * 2;
    // Initialize dynamic programming array.We use dp[i][k][0] for the starting time of k_th microbatch's forward of stage i - 1, dp[i][k][1] for the starting time of k_th microbatch's backward of stage i - 1.
    vector<vector<vector<long long>>> dp(num_pipeline_stages + 2, vector<vector<long long>>(num_microbatches, vector<long long>(2, 0)));
    long long initial_start_time_for_first_backward = 0;
    for(int stage = 0; stage < num_pipeline_stages; stage++) {
        initial_start_time_for_first_backward += forward_time[stage + 1];
        // Add communication overhead except last stage as last stage does not need to send forward to the next stage.
        if(stage != num_pipeline_stages - 1) initial_start_time_for_first_backward += kCommunicationOverhead;
    }
    // Note:from last stage to first stage.
    for(int stage = num_pipeline_stages - 1; stage >= 0; stage--) {
        dp[stage + 1][0][0] = INT32_MIN;
        dp[stage + 1][0][1] = initial_start_time_for_first_backward;
        initial_start_time_for_first_backward += backward_time[stage + 1];
        // Add communication overhead.
        initial_start_time_for_first_backward += kCommunicationOverhead;
    }
    // Get start time for all microbatches in steady phase for every stage.
    for(int num_microbatch = 1; num_microbatch < num_microbatches; num_microbatch++) {
        // Calculate the start time of forward from first stage to last stage.
        for(int stage = 0; stage < num_pipeline_stages; stage++) {
            if(num_microbatch <= last_batch[stage]) {
                dp[stage + 1][num_microbatch][0] = max(dp[stage][num_microbatch - 1][0] + forward_time[stage], dp[stage + 1][num_microbatch - 1][1] + backward_time[stage + 1]);
                // Add communication overhead except first stage as first stage does not need to receive forward from the previous stage.
                if(stage != 0) dp[stage + 1][num_microbatch][0] += kCommunicationOverhead;
            }
        }
        // Calculate the start time of backward from lase stage to first stage.
        for(int stage = num_pipeline_stages - 1; stage >= 0; stage--) {
            if(num_microbatch <= last_batch[stage]) {
                dp[stage + 1][num_microbatch][1] = max(dp[stage + 2][num_microbatch][1] + backward_time[stage + 2], dp[stage + 1][num_microbatch][0] + forward_time[stage + 1]);
                // Add communication overhead except last stage as last stage does not need to receive backward from the next stage.
                if(stage != num_pipeline_stages - 1) dp[stage + 1][num_microbatch][1] += kCommunicationOverhead;
            }
        }
    }
    // We assume the stage of critical path has no bubble during 1F1B steady phase as it dominates the pipeline forward and backward communications.So we can use this feature to find the stage of critical path.And noting that we need to find the stage of critical path from last stage to first stage, because there may be the same forward and backward time for different stages,only the stage close to last stage dominates the pipeline.
    int stage_of_critical_path = num_pipeline_stages - 1;
    while(stage_of_critical_path >= 0) {
        int num_microbatch;
        // As we take communication overhead into consideration during dynamic programming,we need to reconsider stage of critical path.So we add two variables below.For first stage,the forward communication is 0,while for last stage, the backward communication is 0.
        long long forward_communication_overhead = 0, backward_communication_overhead = 0;
        if(stage_of_critical_path != 0) {
            forward_communication_overhead = kCommunicationOverhead;
        }
        if(stage_of_critical_path != num_pipeline_stages - 1) {
            backward_communication_overhead = kCommunicationOverhead;
        }
        for(num_microbatch = 1; num_microbatch <= last_batch[stage_of_critical_path]; num_microbatch++) {
            // Judgement for forward microbatch
            if(dp[stage_of_critical_path + 1][num_microbatch][0] != dp[stage_of_critical_path + 1][num_microbatch - 1][1] + backward_time[stage_of_critical_path + 1] + forward_communication_overhead) break;
            // Judgement for backward microbatch
            if(dp[stage_of_critical_path + 1][num_microbatch][1] != dp[stage_of_critical_path + 1][num_microbatch][0] + forward_time[stage_of_critical_path + 1] + backward_communication_overhead) break;
        }
        if(num_microbatch == last_batch[stage_of_critical_path] + 1) break;
        stage_of_critical_path--;
    }

    // Processing errors.
    if(stage_of_critical_path < 0){
        cout << "Stage of critical path judgement error." << endl;
        cout << "Forward time array:[";
        for(int i = 1; i <= num_pipeline_stages; i++) {
            cout << forward_time[i] << ",";
        }
        cout << "]" << endl;
        cout << "Backward time array:[";
        for(int i = 1; i <= num_pipeline_stages; i++) {
            cout << backward_time[i] << ",";
        }
        cout << "]" << endl;
        cout << "Dynamic programming array:" << endl;
        for(int pipeline_stage = 1; pipeline_stage <= num_pipeline_stages; pipeline_stage++) {
            cout << "{";
            for(int microbatch = 0; microbatch <= last_batch[pipeline_stage - 1]; microbatch++) {
                cout << "[" << dp[pipeline_stage][microbatch][0] << "," << dp[pipeline_stage][microbatch][1] << "],";
            }
            cout << "}" << endl;
        }
        return make_pair(__LONG_LONG_MAX__, 0);
    }
    // return the start time of last forward microbatch and the corresponding stage.
    return make_pair(dp[stage_of_critical_path + 1][last_batch[stage_of_critical_path]][0], stage_of_critical_path);
}

// CalculationForCooldownPhase function which take ealier stages into
// consideration when calculating the start time.By considering the ealiser stages, we can
// take into account the case of communication congestion.
long long CalculationForCooldownPhase(int& num_pipeline_stages, int& stage_of_critical_path, long long& last_forward_microbatch_start_time, vector<long long>& forward_time, vector<long long>& backward_time) {
    // Initialize the dynamic programming array.
    int vector_size = num_pipeline_stages - stage_of_critical_path;
    long long backward_start_time = last_forward_microbatch_start_time;
    vector<vector<long long>> dp(vector_size, vector<long long>(vector_size, 0));
    for(int i = 0;i < vector_size; i++) {
        backward_start_time += forward_time[stage_of_critical_path + 1 + i];
        // Take communication overhead into consideration.Do not add communication overhead for last stage as it does not send forward to next stage.
        if(stage_of_critical_path + i != num_pipeline_stages - 1) {
            backward_start_time += kCommunicationOverhead;
        }
        // j for first microbatch in cooldown phase.
        int j = vector_size - 1 - i;
        dp[i][j] = backward_start_time;
    }
    // Now running for the dynamic programming.Calculating the start time of last backward on stage of critical path.
    for(int col = vector_size - 2; col >= 0; col--) {
        for(int row = vector_size - col - 2; row >= 0; row--) {
            dp[row][col] = max(dp[row][col + 1] + backward_time[stage_of_critical_path + 1 + row], dp[row + 1][col] + backward_time[stage_of_critical_path + 1 + row + 1]);
            // Add communication overhead during dynamic programming.Add communication overhead for first stage,even though it does not send backward to previous stage,it needs a idle time to receive backward from next stage.
            dp[row][col] += kCommunicationOverhead;

            // Take ealier stages into consideration.By considering the ealiser stages, we can
            // take into account the case of communication congestion.For the stage of critical /// path, we do not take ealier stages into consideration as it dominates them.
            if(row > 0) {
                dp[row][col] = max(dp[row][col], dp[row - 1][col + 1]);
            }
        }
    }
    // return the start time of last backward on stage of cirtical path.
    return dp[0][0];
}

void OutputResult(vector<vector<int>>& block_partition, int& stage_of_critical_path, long long& block_partition_cost) {
    // Output current block partition and time cost.
    cout << "------------------------------------------------------------" << endl;
    cout << "Num microbatches:" << 2 *  block_partition.size() << "   Num pipeline stages:" << block_partition.size() << endl;
    cout << "Current block partition with cost: " << block_partition_cost << " us is:" << endl;
    // Output the block partition on the screen.
    cout << "[" << endl;
    for(vector<int> partition:block_partition){
        cout << "[";
        for(int& block:partition){
            cout << block + 1 << ",";
        }
        cout << "]," << endl;
    }
    cout << "]" << endl;
    cout << "Stage of critical path is: " << stage_of_critical_path << endl;
    cout << "------------------------------------------------------------" << endl;
    return;
}


PYBIND11_MODULE(autopipe, m) {
    m.doc() = "A module of AutoPipe for pipeline partition generating.";
    m.def("pipeline", &MerakPipe, "Generating pipeline partition.");
}
