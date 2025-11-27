"""Simple demo showing how to instantiate models via the central registry."""

from src import available_models, get_model, MODEL_REGISTRY
from src.continuous import AutoencoderKL_f4


def main() -> None:
    print("Available token models:", available_models(""))

    info = MODEL_REGISTRY.get("continuous.autoencoder.kl-f4d3")
    print(info)


    autoencoder = get_model(info.name)
    print(f"Autoencoder type: {autoencoder.__class__.__name__}")

    vq_model = get_model("discrete.vq.f4", img_size=64, dims=3)
    print(f"VQ model codebook size: {vq_model.quantizer.n_e}")

    tokenizer = get_model("token.detok.bb", image_size=64)
    print(f"Tokenizer: {tokenizer.__class__.__name__}")



if __name__ == "__main__":
    main()

