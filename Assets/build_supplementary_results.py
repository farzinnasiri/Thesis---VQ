"""Build the thesis results appendix from the archived aggregate CSV files."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Chapters/app_c_supporting_results.tex"
MODEL_ORDER = {"llamagen": 0, "vqgan": 1}
MODE_ORDER = {"closest": 0, "orthogonal": 1, "random_uniform": 2, "farthest": 3}
DATASETS = {"imagenet_val": "ImageNet validation", "imagenetv2": "ImageNet-V2"}


def rows(relative_path):
    with (ROOT / relative_path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, places=3):
    if value is None or value == "":
        return "--"
    return f"{float(value):.{places}f}"


def model_name(value):
    return "LlamaGen" if value.lower() == "llamagen" else "VQGAN"


def latex_table(caption, label, columns, headers, body, font_size="small"):
    short_caption = caption.split(". ", 1)[0].rstrip(".")
    result = [
        "\\begingroup",
        f"\\{font_size}",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{longtable}}{{{columns}}}",
        f"\\caption[{short_caption}]{{{caption}}}\\label{{{label}}}\\\\",
        "\\hline",
        " & ".join(headers) + " \\\\ \\hline",
        "\\endfirsthead",
        f"\\multicolumn{{{len(headers)}}}{{l}}{{\\small\\itshape Continued from the previous page}}\\\\",
        "\\hline",
        " & ".join(headers) + " \\\\ \\hline",
        "\\endhead",
        "\\hline",
        "\\endfoot",
    ]
    result.extend(" & ".join(row) + " \\\\" for row in body)
    result.extend(["\\end{longtable}", "\\endgroup", ""])
    return "\n".join(result)


parts = [
    r"\chapter{Supporting Experimental Results}",
    r"\label{app:supporting_results}",
    "",
    "The tables below retain the per-condition summaries behind the figures and",
    "selected comparisons in Chapters~\\ref{ch:rq2}--\\ref{ch:rq5}. A dash marks a",
    "measurement missing from the archived summary, not an estimated value.",
    "The model runs used the same nominal settings but did not save shared random",
    "interventions; cross-model values are descriptive comparisons.",
    "",
    r"\section{Global image noise}",
]

global_rows = sorted(
    rows("Assets/rq2/data/ch05_table_joint_global_robustness_summary.csv"),
    key=lambda row: (MODEL_ORDER[row["model"]], float(row["sigma"])),
)
parts.append(
    latex_table(
        "Complete ImageNet global-noise summary. PSNR values are dataset means in dB; "
        "flip is the fraction of changed token assignments, and top-100 mass is "
        "the assignment probability held by the 100 most frequent codes.",
        "tab:app_global_noise",
        "llrrrrr",
        ["Tokenizer", "$\\sigma$", "$\\operatorname{PSNR}(x,\\hat{x}_{\\sigma})$", "Flip", "Perplexity", "Top-100 mass", "Active codes"],
        [
            [
                model_name(row["model"]),
                number(row["sigma"], 2),
                number(row["psnr_x_xhat_mean"], 2),
                number(row["token_flip_frac_mean"]),
                number(row["perplexity"], 0),
                number(row["top_100_mass"]),
                f'{int(row["active_codes"]):,}'.replace(",", r"{,}"),
            ]
            for row in global_rows
        ],
    )
)

parts.extend([r"\pagebreak", r"\section{Encoder response by distance from the patch}", ""])
distance_rows = rows("Assets/rq3/data/ch06_table_distance_profile.csv")
distance_index = {
    (row["model"], row["sigma"], int(row["distance"])): row
    for row in distance_rows
}
if len(distance_index) != 102:
    raise ValueError("Expected 102 distinct distance-profile conditions")
parts.append(
    latex_table(
        "Token-flip probability by Manhattan distance from the perturbed patch. "
        "Distance zero denotes tokens inside the patch. Every value is taken from "
        "the saved Chapter 6 distance profile; the six columns are model and noise-level pairs.",
        "tab:app_encoder_distance",
        "r|rrr|rrr",
        ["Distance", "VQ 0.1", "VQ 0.25", "VQ 0.5", "Llama 0.1", "Llama 0.25", "Llama 0.5"],
        [
            [str(distance)]
            + [
                number(distance_index[(model, sigma, distance)]["flip_probability"])
                for model in ("vqgan", "llamagen")
                for sigma in ("0.1", "0.25", "0.5")
            ]
            for distance in range(17)
        ],
    )
)

fidelity = rows("Assets/rq4/data/ch07_table_fidelity_summary.csv")
locality = rows("Assets/rq4/data/ch07_table_locality_summary.csv")
for dataset, dataset_title in DATASETS.items():
    parts.extend([r"\section{" + f"Token edits on {dataset_title}" + "}", ""])
    selected_fidelity = sorted(
        (row for row in fidelity if row["dataset"] == dataset),
        key=lambda row: (
            MODEL_ORDER[row["model"]], int(row["fraction_label"]), MODE_ORDER[row["mode"]]
        ),
    )
    selected_locality = sorted(
        (row for row in locality if row["dataset"] == dataset),
        key=lambda row: (
            MODEL_ORDER[row["model"]], int(row["fraction_label"]), MODE_ORDER[row["mode"]]
        ),
    )
    if len(selected_fidelity) != 32 or len(selected_locality) != 32:
        raise ValueError(f"Expected 32 edit modes per metric family for {dataset}")
    table_rows = lambda source, fields: [
        [
            model_name(row["model"]),
            row["fraction_label"] + r"\%",
            "Random" if row["mode"] == "random_uniform" else row["mode"].capitalize(),
        ] + [number(row[field], 2 if field.startswith("psnr") else 3) for field in fields]
        for row in source
    ]
    for scope, fields in (
        ("patch", ["psnr_patch_mean", "ssim_patch_mean", "lpips_patch_mean"]),
        ("full image", ["psnr_full_mean", "ssim_full_mean", "lpips_full_mean"]),
    ):
        parts.append(
            latex_table(
                f"{dataset_title} token-edit fidelity over the {scope}. "
                "Patch fraction is the target fraction of the $16\\times16$ token grid; "
                "PSNR is in dB. All metrics compare the edited reconstruction with the clean reconstruction.",
                f'tab:app_{dataset}_{scope.replace(" ", "_")}_fidelity',
                "lllr rr".replace(" ", ""),
                ["Tokenizer", "Patch", "Edit", "PSNR", "SSIM", "LPIPS"],
                table_rows(selected_fidelity, fields),
            )
        )
    parts.append(
        latex_table(
            f"{dataset_title} decoder locality after token edits. Inside and outside are "
            "mean pixel RMSE relative to the clean reconstruction; leakage is "
            "the median per-image outside-to-inside ratio. The 99th percentile "
            "is the mean across images of each image's outside-patch 99th percentile.",
            f"tab:app_{dataset}_locality",
            "lllrrrr",
            ["Tokenizer", "Patch", "Edit", "Inside", "Outside", "Leakage", "Outside P99"],
            table_rows(
                selected_locality,
                ["inside_change_mean", "outside_change_mean", "leakage_ratio_median", "outside_p99_mean"],
            ),
        )
    )
    fidelity_index = {
        (row["model"], row["fraction_label"], row["mode"]): row
        for row in selected_fidelity
    }
    locality_index = {
        (row["model"], row["fraction_label"], row["mode"]): row
        for row in selected_locality
    }
    if fidelity_index.keys() != locality_index.keys():
        raise ValueError(f"Fidelity/locality conditions differ for {dataset}")
    parts.append(
        latex_table(
            f"{dataset_title} variation across images for each token-edit condition. "
            "Columns are the saved per-condition standard deviations: patch PSNR "
            "in dB, patch LPIPS, and inside and outside pixel RMSE.",
            f"tab:app_{dataset}_variability",
            "lllrrrr",
            ["Tokenizer", "Patch", "Edit", "PSNR SD", "LPIPS SD", "Inside SD", "Outside SD"],
            [
                [
                    model_name(key[0]),
                    key[1] + r"\%",
                    "Random" if key[2] == "random_uniform" else key[2].capitalize(),
                    number(fidelity_index[key]["psnr_patch_std"], 2),
                    number(fidelity_index[key]["lpips_patch_std"]),
                    number(locality_index[key]["inside_change_std"]),
                    number(locality_index[key]["outside_change_std"]),
                ]
                for key in fidelity_index
            ],
        )
    )

parts.extend([r"\section{Distribution-shift artifact inventory}", ""])
inventory = rows("Assets/rq5/data/ch08_artifact_inventory.csv")
inventory.sort(key=lambda row: (MODEL_ORDER[row["model"].lower()], row["dataset_label"]))
if len(inventory) != 14 or any(row["counts_consistent"] != "True" for row in inventory):
    raise ValueError("Unexpected Chapter 8 artifact inventory")
parts.append(
    latex_table(
        "Encoded Chapter 8 artifacts. Every row has a $16\\times16$ token grid "
        "and satisfies total tokens = images $\\times$ 256. Active codes count "
        "indices observed at least once on that dataset.",
        "tab:app_shift_inventory",
        "llrrr",
        ["Tokenizer", "Dataset", "Images", "Total tokens", "Active codes"],
        [
            [
                row["model"], row["dataset_label"],
                f'{int(row["n_images"]):,}'.replace(",", r"{,}"),
                f'{int(row["total_tokens"]):,}'.replace(",", r"{,}"),
                f'{int(row["active_codes"]):,}'.replace(",", r"{,}"),
            ]
            for row in inventory
        ],
    )
)

manifest = rows("Assets/reproducibility_manifest.csv")
parts.extend([r"\section{Reproducibility index}", ""])
parts.append(
    latex_table(
        "Script and notebook revisions in the audited snapshot. Modified means "
        "the saved file differs from the listed commit. The complete SHA-256 "
        "digests and file paths are in the versioned reproducibility manifest.",
        "tab:app_reproducibility_index",
        r"p{0.38\textwidth}p{0.32\textwidth}ll",
        ["Component", "Repository", "Revision", "State"],
        [
            [
                row["component"],
                row["repository"].split("/")[-1].replace("_", r"\_"),
                r"\texttt{" + row["revision"][:8] + "}",
                row["working_tree_state"],
            ]
            for row in manifest
        ],
        font_size="scriptsize",
    )
)

OUTPUT.write_text("\n".join(parts).rstrip() + "\n")
print(f"Wrote {OUTPUT} from {len(manifest)} manifest records and archived chapter summaries")
