from __future__ import annotations

import argparse


def linear_params(input_dim: int, output_dim: int, bias: bool = True) -> int:
    return input_dim * output_dim + (output_dim if bias else 0)


def linear_flops(batch: int, tokens: int, input_dim: int, output_dim: int) -> int:
    return 2 * batch * tokens * input_dim * output_dim


def trainable_params(temporal_layers: int) -> int:
    one_layer_params = 13_432_837
    transformer_layer_params = 3_152_384
    return one_layer_params + (temporal_layers - 1) * transformer_layer_params


def attention_flops(batch: int, length: int, dim: int) -> int:
    return 2 * batch * length * 4 * dim * dim + 4 * batch * length * length * dim


def temporal_flops(batch: int, length: int, layers: int) -> int:
    d = 512
    return layers * (
        attention_flops(batch, length, d)
        + 2 * batch * length * d * (4 * d) * 2
    )


def video_flops(batch: int, length: int) -> dict[str, int]:
    d, hidden = 512, 256
    return {
        "adjacency": 2 * batch * length * length * d,
        "graph_convolution": (
            4 * (2 * batch * length * d * hidden + 2 * batch * length * length * hidden)
            + 2 * (2 * batch * length * d * hidden * 5)
        ),
        "video_linear": linear_flops(batch, length, d, d),
    }


def sie_flops(batch: int, length: int) -> int:
    dim, top_k = 512, max(1, min(length, int(length * 0.17 + 0.999999)))
    return (
        4 * batch * length * dim
        + 2 * batch * length * top_k
        + 2 * batch * length * top_k
        + 4 * batch * length * dim
    )


def fusion_flops(batch: int, length: int) -> int:
    d = 512
    return (
        2 * attention_flops(batch, length, d)
        + 2 * linear_flops(batch, length, d, d)
        + linear_flops(batch, length, 2 * d, d)
        + linear_flops(batch, length, d, d)
    )


def forward_flops(batch: int, length: int, classes: int, temporal_layers: int) -> dict[str, int]:
    d = 512
    result = {"temporal": temporal_flops(batch, length, temporal_layers)}
    result.update(video_flops(batch, length))
    result["sie"] = sie_flops(batch, length)
    result["fusion"] = fusion_flops(batch, length)
    calibration = 2 * (
        linear_flops(batch, length, d, 256)
        + linear_flops(batch, length, 256, d)
    )
    result["lookback_max_3"] = 3 * (calibration + result["sie"] + result["fusion"])
    result["prediction_heads"] = (
        2 * linear_flops(batch, length, d, 4 * d)
        + 2 * linear_flops(batch, length, 4 * d, d)
        + linear_flops(batch, length, d, 1)
        + 2 * batch * length * classes * d
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("xd", "tmvad"), default="xd")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=256)
    args = parser.parse_args()

    classes, layers = (7, 1) if args.dataset == "xd" else (2, 2)
    params = trainable_params(layers)
    flops = forward_flops(args.batch_size, args.length, classes, layers)
    total = sum(flops.values())
    print(f"Dataset: {args.dataset.upper()} (B={args.batch_size}, T={args.length})")
    print(f"Trainable Params: {params:,} ({params / 1e6:.6f} M)")
    for name, value in flops.items():
        print(f"{name:20s}: {value / 1e9:.6f} G")
    print(f"FLOPs (base + max 3 look-back): {total / 1e9:.6f} G")


if __name__ == "__main__":
    main()
