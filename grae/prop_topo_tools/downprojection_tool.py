import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import sys
import os

from configs.global_config import GlobalConfig
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from moviepy import ImageSequenceClip
import matplotlib.gridspec as gridspec

sys.path.append("C:/Users\MiladBassil/Desktop/Master_Thesis/code\Master_Thesis_Code")

from models.topology_models.custom_topo_tools.connectivity_topo_regularizer import (
    TopologicalZeroOrderLoss,
)
from adan_pytorch import Adan
from models.topology_models.topo_tools.moor_topo_reg import TopologicalSignatureDistance

# Define the custom loss function
class DistanceMatrixLoss(nn.Module):
    def forward(self, original_distances, projected_distances):
        return torch.mean((original_distances - projected_distances) ** 2)


# Downprojection tool
class ConnectivityDP:
    def __init__(
        self,
        n_components=2,
        n_iter=100,
        learning_rate=1,
        optimizer_name="sgd",
        normalize_input=False,
        initialization_scheme="random_uniform",
        weight_decay=0.0,
        loss_calculation_timeout=1.0,
        augmentation_scheme={},
        importance_calculation_strat=None,
        take_top_p_scales=1,
        show_progress_bar = True,
        dev_settings={},
    ):
        self.n_iter = n_iter
        self.n_components = n_components
        self.initialization_scheme = initialization_scheme
        self.learning_rate = learning_rate
        self.normalize_input = normalize_input
        self.show_progress_bar = show_progress_bar
        self.loss_calculation_timeout = loss_calculation_timeout
        self.take_top_p_scales = take_top_p_scales
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.augmentation_scheme = augmentation_scheme
        self.importance_calculation_strat = importance_calculation_strat
        self.dev_settings = dev_settings
        method = "deep"
        if "moor_method" in self.dev_settings:
            method = "moor_method"


        if method == "moor_method":
            self.loss_fn = TopologicalSignatureDistance(match_edges = 'symmetric',to_gpu=False)
        else:
            self.loss_fn = TopologicalZeroOrderLoss(
                method=method,
                timeout=self.loss_calculation_timeout,
                take_top_p_scales=self.take_top_p_scales,
                importance_calculation_strat=importance_calculation_strat,
            )
        self.opt_loss = -1.0
        self.opt_embedding = None
        self.opt_epoch = -1

    def fit_transform(self, X):
        """
        Downproject the input nxd space to 2D by minimizing the distance matrix loss.
        Args:
            X: Input tensor of shape (n, d).
        Returns:
            Optimized 2D embedding of shape (n, n_componenets).
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if self.normalize_input:
            X = self.normalize_perfeature_input(X)
        initial_embedding = self.get_initial_embedding(X)
        original_distances = self.calculate_eucl_distance_matrix(X)
        target_embedding = initial_embedding

        optimizer = self.get_optimizer(target_embedding)

        progress_bar = tqdm(range(self.n_iter), desc="CDP Progress", unit="step") if self.show_progress_bar else range(self.n_iter)
        if "create_vid" in self.dev_settings:
            self.create_update_video(initial_embedding, torch.tensor(-1.0), {}, 0)
        for i in progress_bar:
            optimizer.zero_grad()
            projected_distances = self.calculate_eucl_distance_matrix(target_embedding)
            aug_og_distance_matrix = self.augment_distance_matirx(original_distances)
            loss, log = self.loss_fn(aug_og_distance_matrix, projected_distances)
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            if "create_vid" in self.dev_settings:
                self.create_update_video(target_embedding, loss, log, i + 1)
            if self.show_progress_bar:
                progress_bar.set_postfix(
                    {
                        "Loss": loss.item(),
                        "%_calc": log.get("percentage_toporeg_calc_2_on_1", 100.0),
                    }
                )
            if loss.item() < self.opt_loss or self.opt_loss == -1:
                self.opt_loss = loss.item()
                self.opt_embedding = target_embedding.detach().numpy()
                self.opt_epoch = int(i)
                
        if "create_vid" in self.dev_settings:
            self.create_update_video(None, None, None, -1)
        return self.opt_embedding

    def get_initial_embedding(self, X):
        np.random.seed(42)
        if self.initialization_scheme == "PCA":
            pca = PCA(n_components=self.n_components)
            return torch.tensor(
                pca.fit_transform(X.numpy()), dtype=torch.float32, requires_grad=True
            )
        elif self.initialization_scheme == "random_uniform":
            return torch.tensor(
                np.random.uniform(-1, 1, size=(X.shape[0], self.n_components)),
                dtype=torch.float32,
                requires_grad=True,
            )
        elif self.initialization_scheme == "random_gaussian":
            return torch.tensor(
                np.random.normal(loc=0, scale=1, size=(X.shape[0], self.n_components)),
                dtype=torch.float32,
                requires_grad=True,
            )
        else:
            raise ValueError(
                f"The initialization method {self.initialization_scheme} is not supported"
            )

    def calculate_eucl_distance_matrix(self, X):
        # Compute the pairwise distance matrix
        return torch.norm(X[:, None] - X, dim=2, p=2)

    def augment_distance_matirx(self, input_distance_matrix):
        if "name" in self.augmentation_scheme:
            if self.augmentation_scheme["name"] == "uniform":
                high = self.augmentation_scheme["p"] + 1
                low = 1 - self.augmentation_scheme["p"]
                upper_triangle = (
                    torch.rand(input_distance_matrix.shape) * (high - low) + low
                )
                symmetric_matrix = (
                    torch.triu(upper_triangle) + torch.triu(upper_triangle, 1).T
                )
                symmetric_matrix.fill_diagonal_(1)
                return input_distance_matrix * symmetric_matrix
        else:
            return input_distance_matrix

    def normalize_perfeature_input(self, X):
        return (X - X.mean(dim=1, keepdim=True)) / X.std(dim=1, keepdim=True)

    def get_optimizer(self, opt_domain):
        if self.optimizer_name == "SGD" or self.optimizer_name == "sgd":
            return torch.optim.SGD(
                [opt_domain], lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adam" or self.optimizer_name == "ADAM":
            return torch.optim.Adam(
                [opt_domain], lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw" or self.optimizer_name == "adamW" or self.optimizer_name == "ADAMW":
            return torch.optim.AdamW(
                [opt_domain], lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adan"  or self.optimizer_name == "ADAN":
            return Adan(
                [opt_domain],
                lr = self.learning_rate,                  # learning rate (can be much higher than Adam, up to 5-10x)
                betas = (0.02, 0.08, 0.01), # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
                weight_decay = self.weight_decay       # weight decay 0.02 is optimal per author
            )
    def create_update_video(self, target_embedding, loss, log, it):
        if it == -1:
            fps = 10
            frames = sorted(
                [
                    os.path.join(self.imgs_folder_path, f)
                    for f in os.listdir(self.imgs_folder_path)
                    if f.endswith(".png")
                ]
            )
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(
                f"{self.vid_folder_path}vid.mp4", codec="libx264", fps=fps
            )
            return
        if not hasattr(self, "vid_folder_path"):
            from datetime import datetime

            current_time = datetime.now()
            formatted_time = current_time.strftime("%m_%d_%H_%M")
            self.vid_folder_path = (
                GlobalConfig.RESULTS_FOLDER_PATH
                + GlobalConfig.CONNECTIVITY_DP_VID_PATH
                + f"{formatted_time}/"
            )
            self.imgs_folder_path = (
                GlobalConfig.RESULTS_FOLDER_PATH
                + GlobalConfig.CONNECTIVITY_DP_VID_PATH
                + f"{formatted_time}/"
                + "images/"
            )
            if os.path.exists(self.vid_folder_path):
                print(
                    f"WARNING: overwriting old data in folder: {self.vid_folder_path}"
                )
            else:
                os.makedirs(self.vid_folder_path)
                os.makedirs(self.imgs_folder_path)
        self.create_and_save_plot(target_embedding, loss, log, it)

    def create_and_save_plot(self, target_embedding, loss, log, it):
        nb_bins = 10
        embedding = target_embedding.detach().cpu().numpy()
        scale_loss_info = log.get("scale_loss_info_2_on_1", [(0.0, 0.0, 0.0)])
        scales, pull_losses, push_losses = zip(*scale_loss_info)

        # Bin edges for scales
        bins = np.linspace(0, 1, nb_bins)  # Adjust as needed

        # Calculate average pull and push losses for each bin
        bin_indices = np.digitize(scales, bins) - 1
        num_bins = len(bins) - 1
        average_pull_losses = [
            (
                np.mean(
                    [pull_losses[i] for i in range(len(scales)) if bin_indices[i] == j]
                )
                if np.sum(bin_indices == j) > 0
                else 0
            )
            for j in range(num_bins)
        ]
        average_push_losses = [
            (
                np.mean(
                    [push_losses[i] for i in range(len(scales)) if bin_indices[i] == j]
                )
                if np.sum(bin_indices == j) > 0
                else 0
            )
            for j in range(num_bins)
        ]

        # Create the figure and GridSpec
        fig = plt.figure(figsize=(16, 8))
        spec = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

        # Scatter plot (big plot on the left)
        ax_scatter = fig.add_subplot(spec[:, 0])
        marker_styles = ['o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'X', 'h', 'H', '8', '|', '_', '.', ',']
        colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors  # Combines multiple color maps for 60+ colors

        # Ensure you have enough unique combinations of colors and markers
        num_classes = len(set(self.dev_settings["labels"]))  # Total number of classes
        unique_labels = sorted(set(self.dev_settings["labels"]))  # Sort labels for consistent ordering
        if num_classes > len(colors) * len(marker_styles):
            raise ValueError("Not enough combinations of colors and markers for all classes!")

        # Create scatter plot with combined color and marker coding
        for idx, label in enumerate(unique_labels):
            color = colors[idx % len(colors)]
            marker = marker_styles[idx // len(colors) % len(marker_styles)]
            subset = embedding[np.array(self.dev_settings["labels"]) == label]
            ax_scatter.scatter(
                subset[:, 0],
                subset[:, 1],
                color=color,
                marker=marker,
                label=f"{label}",
                s=50,
            )

        # Add legend for clarity
        ax_scatter.legend(
            title="Classes",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
        )
        # scatter = ax_scatter.scatter(
        #     embedding[:, 0],
        #     embedding[:, 1],
        #     c=self.dev_settings["labels"],
        #     cmap="tab10",
        #     s=50,
        # )
        ax_scatter.set_title("Labeled Embedding")
        ax_scatter.set_xlabel("Component 1")
        ax_scatter.set_ylabel("Component 2")
        #plt.colorbar(scatter, ax=ax_scatter, label="Label")

        # First histogram: Counts in each scale bin
        ax_hist1 = fig.add_subplot(spec[0, 1])
        ax_hist1.hist(scales, bins=bins, color="gray", edgecolor="black")
        ax_hist1.set_title("Number of Persistent Edges in Each Scale Bracket")
        ax_hist1.set_xlabel("Scale")
        ax_hist1.set_ylabel("Count")

        # Second histogram: Average losses
        ax_hist2 = fig.add_subplot(spec[1, 1])
        bar_width = 0.35
        x = np.arange(num_bins)
        ax_hist2.bar(
            x - bar_width / 2,
            average_pull_losses,
            width=bar_width,
            color="blue",
            label="Pull Loss",
        )
        ax_hist2.bar(
            x + bar_width / 2,
            average_push_losses,
            width=bar_width,
            color="red",
            label="Push Loss",
        )
        ax_hist2.set_xticks(x)
        ax_hist2.set_xticklabels(
            [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(num_bins)], rotation=45
        )
        ax_hist2.set_title("Average Losses in Scale Bracket")
        ax_hist2.set_xlabel("Scale Bin")
        ax_hist2.set_ylabel("Average Loss")
        ax_hist2.legend()

        fig.subplots_adjust(top=0.9)
        fig.suptitle(
            f"Dataset: {self.dev_settings.get('dataset_name','Unknown')} ,Iteration {it}, Loss: {loss.item():.4f}, Iteration Time: {float(log.get('topo_time_taken_2_on_1', -1)):.4f} s, Total Pull Loss: {sum(pull_losses):.4f}, Total Push Loss: {sum(push_losses):.4f}, Loss Percentage Calculated: {log.get('percentage_toporeg_calc_2_on_1', 100):.3f}%",
            fontsize=12,
        )

        # Save and close the figure
        plt.tight_layout()
        plt.savefig(f"{self.imgs_folder_path}/frame_{it:05d}.png")
        plt.close()


    def write_iteration_data(
        file_path, iteration, embedding, loss, histogram, heatmap_stats
    ):
        """Append iteration data to a JSON file in JSON Lines format."""
        # Construct the data to save (ensure minimal size)
        data = {
            "iteration": iteration,
            "embedding": [round(float(e), 4) for e in embedding],  # truncate precision
            "loss": round(float(loss), 6),
            "histogram": histogram,  # e.g., bins or summary stats
            "heatmap": heatmap_stats,  # e.g., min, max, mean values
        }

        # Append to the JSON Lines file
        with open(file_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def read_iteration_data(file_path):
        """Read data line-by-line from a JSON Lines file."""
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)  # Deserialize line into a dictionary
                yield data

if __name__ == "__main__":
    n, d = 100, 10
    X = np.random.rand(n, d)
    tool = ConnectivityDP(n_iter=100, learning_rate=0.01)
    embedding = tool.fit_transform(X)
    import matplotlib.pyplot as plt

    plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.8)
    plt.title("Downprojected Embedding")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
