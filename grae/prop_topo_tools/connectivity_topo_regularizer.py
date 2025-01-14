
import torch.nn as nn
from torch import stack,tensor,Tensor,long,abs
from numpy import ndarray
import numpy as np
import time
from random import shuffle
from concurrent.futures import ProcessPoolExecutor,as_completed,TimeoutError
from math import ceil,floor

from grae.prop_topo_tools.topo_encoder import ConnectivityEncoderCalculator

class Timer:
    def __init__(self, timeout=2, start_time = None):
        self.timeout = timeout
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
    def clear(self):
        self.start_time = time.time()
    def set(self):
        pass #this is nothing, just to mkae it compatible with mutithreading approach
    def is_set(self):
        """Check if the timer has exceeded the set timeout."""
        return (time.time() - self.start_time) > self.timeout
        
    def __getstate__(self):
        # Prepare the state dictionary, including start_time and timeout
        return {'timeout': self.timeout, 'start_time': self.start_time}

    def __setstate__(self, state):
        # Restore the object's state exactly as it was
        self.timeout = state['timeout']
        self.start_time = state['start_time']
    
def deep_topo_loss_at_scale(topo_encoding_space_1,topo_encoding_space_2,s1_scale_indices,stop_event):
        scale_edges_pairings = []
        for s1_scale_index in s1_scale_indices:
            if not stop_event.is_set():
                assert isinstance(s1_scale_index,int)
                assert 0<= s1_scale_index< len(topo_encoding_space_1.scales)
                component_birth_in_s1_due_to_pers_pair = topo_encoding_space_1.get_component_birthed_at_index(s1_scale_index)
                scale_in_s1 = topo_encoding_space_1.scales[s1_scale_index]
                index_of_scale_in_s2 = topo_encoding_space_2.get_index_of_scale_closest_to(scale_in_s1)
                relevant_sets_in_s2 = topo_encoding_space_2.get_components_that_contain_these_points_at_this_scale_index(
                    relevant_points=component_birth_in_s1_due_to_pers_pair, index_of_scale=index_of_scale_in_s2 
                )
                to_push_out_at_this_scale = []
                healthy_subsets = {}

                for component_in_s2_name, member_vertices in relevant_sets_in_s2.items():
                    good_vertices = np.intersect1d(member_vertices, component_birth_in_s1_due_to_pers_pair)
                    for vertex in member_vertices:
                        if vertex not in component_birth_in_s1_due_to_pers_pair:
                            pair_info = topo_encoding_space_2.what_connected_this_point_to_this_set(
                                point=vertex, vertex_set=good_vertices
                            )["persistence_pair"]
                            to_push_out_at_this_scale.append(pair_info)
                    healthy_subsets[component_in_s2_name] = good_vertices#tensor(good_vertices, dtype=long, device=distances2.device)
                if len(healthy_subsets) > 1:
                    pairs_to_pull = topo_encoding_space_2.what_edges_needed_to_connect_these_components(healthy_subsets)
                else:
                    pairs_to_pull = []
                unique_to_push_out_at_this_scale = list(set(to_push_out_at_this_scale))
                scale_edges_pairings.append((scale_in_s1, pairs_to_pull,unique_to_push_out_at_this_scale,s1_scale_index))
            else:
                return scale_edges_pairings
        return scale_edges_pairings


class TopologicalZeroOrderLoss(nn.Module):
    """Topological signature."""
    LOSS_ORDERS = [1,2]
    PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS =["match_scale_order","match_scale_distribution","moor_method","modified_moor_method","deep"]

    def __init__(self,method="deep",p=2,timeout = 5,multithreading = True,importance_scale_fraction_taken=1.0,importance_calculation_strat = None):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        assert p in TopologicalZeroOrderLoss.LOSS_ORDERS
        self.p = p
        self.signature_calculator = ConnectivityEncoderCalculator
        self.loss_fnc = self.get_torch_p_order_function()
        self.topo_feature_loss = self.get_topo_feature_approach(method)
        self.importance_scale_fraction_taken = importance_scale_fraction_taken
        self.calculate_all_losses = False
        self.importance_calculation_strat = importance_calculation_strat
        self.timeout = timeout
        self.multithreading= multithreading
        if self.multithreading and method == "deep":
            self.available_threads = floor(self.get_thread_count() * 0.5) #take up 50% of available threads excluding the current one or the one this is executing on
            if self.available_threads == 0:
                self.multithreading = False
                self.stop_event = Timer(self.timeout)
            else:
                self.executor =  ProcessPoolExecutor(max_workers=self.available_threads)#ThreadPoolExecutor(max_workers=self.available_threads)#
                self.stop_event = Timer(self.timeout) #threading.Event()
                self.main_thread_event = Timer(self.timeout*0.9)
                try:
                    import wandb
                    wandb.run.summary["threads_used_for_topo_calc"] = self.available_threads
                except Exception as e:
                    print(f"Could not log threads used by deep topology regularization, probably because the loss is being used standalone, available threads: {self.available_threads}; error:{e}")
        else:
            self.available_threads = 0
            self.multithreading = False
            self.stop_event = Timer(self.timeout)
        print(f"Available threads : {self.available_threads}")
    
    def get_top_p_indices(self,topo_encoding_space: ConnectivityEncoderCalculator):
        n_top = int(len(topo_encoding_space.component_total_importance_score) * self.importance_scale_fraction_taken)
        values_array = np.array(topo_encoding_space.component_total_importance_score)

        sorted_indices = np.argsort(values_array)[::-1]

        cutoff_value = values_array[sorted_indices[n_top - 1]]
        eligible_indices = [i for i, v in enumerate(topo_encoding_space.component_total_importance_score) if v == cutoff_value]

        good_indices = [i for i, v in enumerate(topo_encoding_space.component_total_importance_score) if v > cutoff_value]
        if len(eligible_indices) > n_top:
            random_subsmaple_of_eligible_indices = random.sample(eligible_indices, n_top - len(good_indices))
        else:
            random_subsmaple_of_eligible_indices = eligible_indices
        good_indices.extend(random_subsmaple_of_eligible_indices)
        return list(good_indices)
        
        

   
    def deep_scale_distribution_matching_loss_of_s1_on_s2(self, topo_encoding_space_1: ConnectivityEncoderCalculator, distances1, topo_encoding_space_2: ConnectivityEncoderCalculator, distances2):
        if distances2.requires_grad or self.calculate_all_losses:            
            self.stop_event.clear()
            nb_of_persistent_pairs = len(topo_encoding_space_1.persistence_pairs)
            shuffled_indices_of_topo_features = self.get_top_p_indices(topo_encoding_space_1)
            shuffle(shuffled_indices_of_topo_features)
            k, m = divmod(len(shuffled_indices_of_topo_features), self.available_threads + 1)
            subdivided_list = [shuffled_indices_of_topo_features[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(self.available_threads + 1)]
            important_edges_for_each_scale = []
            start_time = time.time()

            if self.multithreading:
                args = [(topo_encoding_space_1,topo_encoding_space_2,indices,self.stop_event) for indices in subdivided_list[:-1]]
                futures = {self.executor.submit(deep_topo_loss_at_scale, *arg): i for i, arg in enumerate(args)}
                self.main_thread_event.clear()
                main_thread_execution_result =  deep_topo_loss_at_scale(topo_encoding_space_1,topo_encoding_space_2,subdivided_list[-1],self.main_thread_event)
                self.stop_event.set()#this us useful if using threading.event structure
                completed = 0
                threads_that_returned = 0
                nb_of_complete_per_thread = []
                try:
                    for future in as_completed(futures,timeout=self.timeout):# this extra timeout is the amount of extra time it will w8 for last iteration to finish, otherwise it will dump all the results
                        try:
                            result = future.result()
                            important_edges_for_each_scale.extend(result)
                            completed = completed + len(result)
                            nb_of_complete_per_thread.append(len(result))
                            threads_that_returned = threads_that_returned + 1
                        except Exception as e:
                            print(f"Error occurred while getting result: {e}")
                except TimeoutError:
                    print(f"Called for stopping of threads, however {len(futures) - threads_that_returned} out of {len(futures) + 1} have failed to terminate within the timeout window of {self.timeout}sec. Throwing results of failed threads and continuing.")
                important_edges_for_each_scale.extend(main_thread_execution_result)
                completed = completed + len(main_thread_execution_result)
                nb_of_complete_per_thread.append(len(main_thread_execution_result))
                std_of_workload_across_threads = np.std(nb_of_complete_per_thread,ddof=1)/np.sum(nb_of_complete_per_thread) if len(nb_of_complete_per_thread)>1 else 0.0
            else:
                important_edges_for_each_scale = deep_topo_loss_at_scale(topo_encoding_space_1,topo_encoding_space_2,subdivided_list[0],self.stop_event)
                completed = len(important_edges_for_each_scale)
                std_of_workload_across_threads = 0.0 


            pairwise_distances_influenced = 0
            nb_pulled_edges = 0
            nb_pushed_edges = 0
            set_of_unique_edges_influenced = set()
            push_loss = tensor(0.0, device=distances2.device)
            pull_loss = tensor(0.0, device=distances2.device)
            scale_demographic_infos = []
            for scale,pull_edges,push_edges,scale_index in important_edges_for_each_scale:
                all_edges = pull_edges + push_edges
                set_of_unique_edges_influenced.update(set(all_edges))
                if len(all_edges) == 0:
                    continue
                push_important_pairs_tensor = tensor(np.array(push_edges), dtype=long, device=distances2.device)
                pull_important_pairs_tensor = tensor(np.array(pull_edges), dtype=long, device=distances2.device)
                scale_demographic_info = [scale,0.0,0.0] #scale,pull,push
                scale = tensor(scale, device=distances2.device)
                if len(pull_edges) != 0:
                    pull_selected_diff_distances = distances2[pull_important_pairs_tensor[:, 0], pull_important_pairs_tensor[:, 1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
                    pull_loss_at_this_scale = abs(pull_selected_diff_distances - scale) ** self.p
                    pull_loss_at_this_scale = pull_loss_at_this_scale.sum()
                    scale_demographic_info[1] = float(pull_loss_at_this_scale.item())
                    pull_loss = pull_loss + pull_loss_at_this_scale*topo_encoding_space_1.component_total_importance_score[scale_index]                     
                if len(push_edges) != 0:
                    push_selected_diff_distances = distances2[push_important_pairs_tensor[:, 0], push_important_pairs_tensor[:, 1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
                    push_loss_at_this_scale = abs(push_selected_diff_distances - scale) ** self.p
                    push_loss_at_this_scale = push_loss_at_this_scale.sum()
                    scale_demographic_info[2] = float(push_loss_at_this_scale.item())
                    push_loss = push_loss + push_loss_at_this_scale*topo_encoding_space_1.component_total_importance_score[scale_index]

                scale_demographic_infos.append(scale_demographic_info)
                pairwise_distances_influenced = pairwise_distances_influenced + len(all_edges)
                nb_pulled_edges = nb_pulled_edges + len(pull_edges)
                nb_pushed_edges = nb_pushed_edges + len(push_edges)
            total_time_section = time.time() - start_time
            #print(f"Total time take for topology calculatation {total_time_section:.4f} seconds, nb of pers_pairs: {nb_of_persistent_pairs} of which {completed} where calculated, with {pairwise_distances_influenced} paris influenced ")
            if pairwise_distances_influenced > 0:
                loss = (push_loss + pull_loss) / completed if completed != 0 else tensor(0.0, device=distances2.device)
                topo_step_stats = {"topo_time_taken": float(total_time_section),"nb_of_persistent_edges":nb_of_persistent_pairs,
                                   "percentage_toporeg_calc":100*float(completed/ nb_of_persistent_pairs),"pull_push_ratio":float(nb_pulled_edges/(0.01+nb_pushed_edges)),
                                   "nb_pairwise_distance_influenced":pairwise_distances_influenced,"nb_unique_pairwise_distance_influenced":len(set_of_unique_edges_influenced),
                                   "rate_of_scale_calculation":float(completed)/float(total_time_section), "pull_push_loss_ratio":pull_loss.item()/push_loss.item() if  push_loss.item() != 0 else -1.0,
                                   "scale_loss_info":scale_demographic_infos,"std_of_workload_across_threads":std_of_workload_across_threads}
            else:
                loss = tensor(0.0, device=distances2.device)
                topo_step_stats = {"topo_time_taken": float(total_time_section),"nb_of_persistent_edges":nb_of_persistent_pairs,
                                   "percentage_toporeg_calc":100*float(completed/ nb_of_persistent_pairs),
                                   "nb_pairwise_distance_influenced":pairwise_distances_influenced,"nb_unique_pairwise_distance_influenced":len(set_of_unique_edges_influenced)}
                
            return loss ,topo_step_stats
        else:
            return tensor(0.0, device=distances2.device),{}
    

    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        nondiff_distances1 = TopologicalZeroOrderLoss.to_numpy(distances1)
        nondiff_distances2 = TopologicalZeroOrderLoss.to_numpy(distances2)
        topo_encoding_space_1 = self.calulate_space_connectivity_encoding(nondiff_distances1)
        topo_encoding_space_2 = self.calulate_space_connectivity_encoding(nondiff_distances2)
        
        loss_2_on_1 = self.topo_feature_loss(topo_encoding_space_1=topo_encoding_space_1,
                                                      distances1=distances1,
                                                      topo_encoding_space_2=topo_encoding_space_2,
                                                      distances2=distances2)
        
        loss_1_on_2 = self.topo_feature_loss(topo_encoding_space_1=topo_encoding_space_2,
                                                      distances1=distances2,
                                                      topo_encoding_space_2=topo_encoding_space_1,
                                                      distances2=distances1)
        loss,log = TopologicalZeroOrderLoss.combine_topo_feature_loss_function_outputs(loss_1_on_2,loss_2_on_1)

        return loss,log
    @staticmethod
    def combine_topo_feature_loss_function_outputs(topo_loss_1_on_2,topo_loss_2_on_1):
        topo_loss_1_on_2 = TopologicalZeroOrderLoss.extract_topo_feature_loss_function_output(topo_loss_1_on_2)
        topo_loss_2_on_1 = TopologicalZeroOrderLoss.extract_topo_feature_loss_function_output(topo_loss_2_on_1)
        log = {f"{key}_1_on_2":value for key,value in topo_loss_1_on_2[1].items()}
        log.update({f"{key}_2_on_1":value for key,value in topo_loss_2_on_1[1].items()})
        log["topo_loss_1_on_2"] =float(topo_loss_1_on_2[0].item())
        log["topo_loss_2_on_1"] =float(topo_loss_2_on_1[0].item())

        combined_loss = topo_loss_1_on_2[0] + topo_loss_2_on_1[0]
        return combined_loss,log
    @staticmethod
    def extract_topo_feature_loss_function_output(topo_output):
        if isinstance(topo_output, tuple):
            log_info = topo_output[1]
            loss = topo_output[0]
        else:
            log_info = {}
            loss = topo_output
        return loss,log_info 

    def get_topo_feature_approach(self,method):
        self.method = self.set_scale_matching_method(method)
        if self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[0]:
            topo_fnc =  self.scale_order_matching_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[1]:
            topo_fnc =  self.scale_distribution_matching_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[2]:
            topo_fnc =  self.moor_method_calculate_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[3]:
            topo_fnc =  self.modified_moor_method_calculate_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[4]:
            topo_fnc = self.deep_scale_distribution_matching_loss_of_s1_on_s2
        print(f"Using {self.method} to calculate per topo feature loss")
        return topo_fnc

    def set_scale_matching_method(self,scale_matching_method):
        if scale_matching_method in TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS:
            return scale_matching_method
        else:
            raise ValueError(f"Scale matching methode {scale_matching_method} does not exist")
        
    def calulate_space_connectivity_encoding(self,distance_matrix):
        topo_encoder = self.signature_calculator(distance_matrix,importance_calculation_strat=self.importance_calculation_strat)
        topo_encoder.calculate_connectivity()
        return topo_encoder

    # using numpy to make sure autograd of torch is not disturbed
    @staticmethod
    def to_numpy(obj):
        if isinstance(obj, ndarray):
            # If it's already a NumPy array, return as is
            return obj
        elif isinstance(obj, Tensor):
            # Detach the tensor from the computation graph, move to CPU if necessary, and convert to NumPy
            return obj.detach().cpu().numpy()
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor")
        
    def moor_method_calculate_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []

        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]]
            scale_of_edge_in_space_2 = distances2[topo_encoding_space_1.persistence_pairs[index][0],topo_encoding_space_1.persistence_pairs[index][0]]
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    def modified_moor_method_calculate_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []

        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]]
            scale_of_edge_in_space_2 = distances2[topo_encoding_space_1.persistence_pairs[index][0],topo_encoding_space_1.persistence_pairs[index][0]]
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)/ topo_encoding_space_2.distance_of_persistence_pairs[-1]
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    
    def scale_order_matching_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []

        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            equivalent_feature_in_space_2 = topo_encoding_space_2.what_connected_these_two_points(edge_indices[0], edge_indices[1])
            equivalent_edge_in_space_2 = equivalent_feature_in_space_2["persistence_pair"]
            scale_of_edge_in_space_1 = tensor(topo_encoding_space_2.scales[index]).to(distances1.device) #will break numpy inputs; and is non differentiable
            scale_of_edge_in_space_2 = distances2[equivalent_edge_in_space_2[0], equivalent_edge_in_space_2[1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]            
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    
    def scale_distribution_matching_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []
        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            equivalent_feature_in_space_2 = topo_encoding_space_2.what_connected_these_two_points(edge_indices[0], edge_indices[1])
            equivalent_edge_in_space_2 = equivalent_feature_in_space_2["persistence_pair"]
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]] / topo_encoding_space_1.distance_of_persistence_pairs[-1]
            scale_of_edge_in_space_2 = distances2[equivalent_edge_in_space_2[0], equivalent_edge_in_space_2[1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    


    
    def get_torch_p_order_function(self):
        if self.p ==1 :
            return nn.L1Loss()
        elif self.p == 2:
            return nn.MSELoss()
        else:
            raise ValueError(f"This loss {self.p} is not supported")
        
    def get_thread_count(self):
        import os
        import subprocess
        import psutil
        
        def get_allocated_threads_by_slurm():
            try:
                # Get the number of threads per core from lscpu
                lscpu_output = subprocess.check_output("lscpu | awk '/Thread\\(s\\) per core:/ {print $4}'", shell=True)
                threads_per_core = int(lscpu_output.decode().strip())
                
                # Get the Slurm allocated CPUs
                cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", None))  # Default to 1 if not in Slurm
                allocated_threads = threads_per_core * cpus_per_task
                return allocated_threads
            except Exception as e:
                print(f"Could not detect SLURM environment, probably running on personal hardware; Error msg {e}")
                return None

        def get_threads_from_hardware():
                try:
                    # Get total number of logical processors (threads)
                    threads = psutil.cpu_count(logical=True)
                    return threads
                except Exception as e:
                    print(f"Using only one thread to calculate topology, could not detect how many threads available; Error: {e}")
                    return 1 #assume only one thread exists
                
        threads_alloc_by_slurm = get_allocated_threads_by_slurm()
        if threads_alloc_by_slurm is None:
            threads_available_on_hardware = get_threads_from_hardware()
            return threads_available_on_hardware
        return get_allocated_threads_by_slurm()