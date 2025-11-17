# my_project/main.py
#%% 
def main():
    print("Starting 3D ResNet Autoencoder Training with wandb...")
    # Import inside main to avoid circular imports
    from trainer.train import train_

    train_()

    print("Training complete!")

if __name__ == "__main__":
    main()

# %%
