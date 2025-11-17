
#%%  
def main():
    print("Starting 3D ResNet Autoencoder Training with wandb...")
    # Import inside main to avoid circular imports
    from trainer.train_direcctInverse import train_convnext_encoder_coeff_predictor

    train_convnext_encoder_coeff_predictor()

    print("Training complete!")

if __name__ == "__main__":
    main()

