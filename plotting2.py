import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

name_mapper = {
    "bellman_ford": "Bellman-Ford",
    "bfs": "BFS",
    "dfs": "DFS",
    "dag_shortest_paths": "DAG shortest paths",
    "floyd_warshall": "Floyd-Warshall",
    "mst_prim": "MST Prim",
    "strongly_connected_components_local": "SCC",
    "binary_search": "Binary Search",
    "insertion_sort": "Insertion Sort",
    "minimum": "Minimum",
}
NOTGRAPH = ["Insertion Sort", "Minimum", "Binary Search"]


run_names_baselines = [
    "decent-bee-661",
    "glowing-valley-660",
    "super-terrain-659",
    "still-sky-644",
    "wise-pine-643",
    "misunderstood-morning-644",
    "earnest-haze-632",
    "rosy-pond-631",
    "scarlet-vortex-630",
    "comic-shadow-620",
    "firm-river-616",
    "wandering-armadillo-615",
    "chocolate-wind-596",
    "comfy-snowball-587",
    "decent-firefly-580",
    "earthy-microwave-579",
    "devout-rain-578",
    "trim-dragon-577",
    "divine-salad-574",
    "fearless-feather-571",
    "floral-glitter-570",
    "expert-water-569",
    "pious-deluge-566",
    "misty-glitter-565",
    "faithful-sky-534",
    "lilac-lion-529",
    "firm-pond-528",
    "unique-dream-522",
    "prime-river-521",
    "fluent-wind-516",
]  # baselines

run_names_deqs = [
    "woven-fire-666",
    "stoic-hill-665",
    "spring-lion-662",
    "stilted-plant-639",
    "comfy-serenity-637",
    "colorful-dust-636",
    "summer-thunder-621",
    "fearless-frog-619",
    "brisk-feather-618",
    "soft-oath-539",
    "northern-sunset-538",
    "wild-disco-537",
    "fallen-glade-448",
    "lilac-dew-446",
    "vivid-haze-445",
    "stellar-firebrand-429",
    "prime-pine-414",
    "brisk-pyramid-413",
    "hopeful-grass-475",
    "balmy-morning-341",
    "trim-thunder-332",
    "jumping-dew-362",
    "clean-pine-363",
    "dazzling-haze-361",
    "glad-plasma-344",
    "prime-capybara-343",
    "confused-feather-342",
    "unique-totem-609",
    "distinctive-vortex-604",
    "different-violet-603",
]  # unmodified DEARs


run_names_deqs_granolated = [
    "feasible-yogurt-1046",
    "happy-galaxy-1037",
    "rich-firefly-1036",
    "glamorous-smoke-1045",
    "curious-snowball-1035",
    "feasible-aardvark-1034",
    "devout-disco-1042",
    "fragrant-butterfly-1028",
    "hearty-shape-1029",
    "toasty-cosmos-1048",
    "peach-cloud-1022",
    "robust-galaxy-1023",
    "earthy-water-1044",
    "comfy-leaf-1019",
    "legendary-thunder-1018",
    "stilted-rain-1047",
    "soft-snow-1011",
    "worthy-gorge-1010",
    "celestial-night-1040",
    "woven-wind-1003",
    "dainty-donkey-1002",
    "fragrant-plasma-1049",
    "faithful-dust-997",
    "firm-frost-996",
    "rare-wildflower-1041",
    "effortless-valley-991",
    "prime-voice-990",
    "bright-snowball-1043",
    "tough-firebrand-980",
    "flowing-pine-979",
]

run_names_deq_cgp = [
    "lucky-dawn-850",
    "wobbly-glitter-849",
    "denim-elevator-848",
    "vital-sponge-820",
    "wobbly-lake-819",
    "hearty-bee-818",
    "different-brook-614",
    "toasty-armadillo-613",
    "glamorous-dust-612",
    "upbeat-meadow-559",
    "lemon-darkness-558",
    "tough-silence-557",
    "chocolate-leaf-697",
    "dazzling-cosmos-477",
    "copper-blaze-476",
    "dashing-lion-465",
    "misty-yogurt-464",
    "decent-darkness-463",
    "classic-pyramid-430",
    "rich-valley-424",
    "bumbling-salad-423",
]


api = wandb.Api()


# Project is specified by <entity/project-name>
def get_df_for_run_names(run_names, whattoget="train/loss/average_loss_epoch"):
    runs = api.runs("clrs-cambridge/nardeq")
    runs = list(filter(lambda x: x.name in run_names, runs))
    hist_list_step = []
    hist_list_epoch = []
    for run in runs:
        name = run.config["algorithm_names"][0]

        hist_epoch = run.history(keys=["_step", "epoch", whattoget])
        hist_epoch["name"] = name_mapper[name]
        hist_list_epoch.append(hist_epoch)

    df = pd.concat(hist_list_epoch, ignore_index=True)
    return df


df_baselines = get_df_for_run_names(run_names_baselines)
df_deq = get_df_for_run_names(run_names_deqs)
df_deq_cgp = get_df_for_run_names(run_names_deq_cgp)
df_deq_granolated = get_df_for_run_names(run_names_deqs_granolated)
df_mlsf = get_df_for_run_names(run_names_deqs, whattoget="val/loss/min_loss_so_far")
df_mlsf_granolated = get_df_for_run_names(
    run_names_deqs_granolated, whattoget="val/loss/min_loss_so_far"
)

epoch_loss = (
    df_deq.groupby(["name", "epoch"])["train/loss/average_loss_epoch"]
    .agg(["mean", "std"])
    .reset_index()
)

sns.set_style("darkgrid")
label_dict = {}

plt.figure(figsize=(10, 6))


# Plot the mean values
for name, group in epoch_loss.groupby("name"):
    plt.plot(group["epoch"], group["mean"], label=label_dict.get(name, name))
    label_dict[name] = None  # Mark the label as used

# Plot the standard deviation as shaded regions
for name, group in epoch_loss.groupby("name"):
    plt.fill_between(
        group["epoch"],
        np.maximum(group["mean"] - group["std"], 1e-4),
        group["mean"] + group["std"],
        alpha=0.2,
        label=label_dict.get(name, name),
    )  # Reuse the label
    label_dict[name] = None  # Mark the label as used

# Customize the plot
plt.tick_params(axis="y", which="both", direction="in", labelsize=20)
plt.tick_params(axis="x", which="both", direction="in", labelsize=20)
plt.xlabel("Epoch", fontsize=24)
plt.ylabel("Train Loss", fontsize=24)
plt.ylim(bottom=1e-5, top=6)
plt.yscale("log")
plt.legend(ncol=2, fontsize=11)
plt.title("DEAR", fontsize=24)

plt.savefig(
    "DEAR_numbers_plot.png", dpi=900, bbox_inches="tight"
)  # Adjust dpi as needed

# def plot_algorithm_loss(ax, df, algorithm_name, label='NAR', solidlabel='NAR', whattoplot='train/loss/average_loss_epoch'):
#     linestyle = '-' if label == solidlabel else '--'
#     algorithm_df = df[df['name'] == algorithm_name]
#     epoch_loss = algorithm_df.groupby('epoch')[whattoplot].agg(['mean', 'std']).reset_index()

#     ax.plot(epoch_loss['epoch'], epoch_loss['mean'], label=label, linestyle=linestyle, color='tab:blue')
#     ax.fill_between(epoch_loss['epoch'],
#                      np.maximum(epoch_loss['mean'] - epoch_loss['std'], 1e-4),
#                      epoch_loss['mean'] + epoch_loss['std'],
#                      alpha=0.2, color='tab:blue')  # Fill between NAR std
#     ax.set_yscale('log')
#     ax.tick_params(axis='both', which='both', direction='in', labelsize=20)
#     ax.set_xlabel('Epoch', fontsize=24)
#     plt.ylabel('Train Loss', fontsize=24)
#     ax.set_ylim(bottom=1e-5, top=6)
#     ax.legend(fontsize=15)

# # Plot and save each algorithm separately
# for algorithm_name in name_mapper.values():
#     if algorithm_name in NOTGRAPH:
#         continue
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # plot_algorithm_loss(ax, df_baselines, algorithm_name)
#     plot_algorithm_loss(ax, df_deq, algorithm_name, label='DEAR', solidlabel='DEAR')
#     plot_algorithm_loss(ax, df_deq_cgp, algorithm_name, label='DEAR w/ CGP')
#     ax.set_title(algorithm_name, fontsize=24)

#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.3)
#     plt.savefig(f'{algorithm_name}_CGP_plot.png', dpi=900, bbox_inches='tight')  # Adjust dpi as needed
#     print('saving', f'{algorithm_name}_CGP_plot.png')
#     plt.close()
