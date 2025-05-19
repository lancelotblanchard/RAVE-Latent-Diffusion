#### RAVE-Latent Diffusion
#### https://github.com/moiseshorta/RAVE-Latent-Diffusion
####
#### Author: Moisés Horta Valenzuela / @hexorcismos
#### Year: 2023

import argparse
import torch.multiprocessing as mp
import torch
import os
import time
import datetime
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb
from librosa.feature import zero_crossing_rate, rms
import numpy as np


from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
else:
    device = torch.device("cpu")
current_date = datetime.date.today()

# We need to project our embedding (ZCR or RMS) to a higher dimension
class EmbeddingProjection(torch.nn.Module):
    def __init__(self, diffusion_model, embedding_projection_size):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.projection0 = torch.nn.Linear(1, embedding_projection_size)
        self.projection1 = torch.nn.Linear(embedding_projection_size, embedding_projection_size)

    def forward(self, *args, embedding=None, **kwargs):
        if embedding is not None:
            embedding = F.silu(self.projection1(F.silu(self.projection0(embedding))))
        return self.diffusion_model(*args, embedding=embedding, **kwargs)

    @torch.no_grad()
    def sample(self, *args, embedding=None, **kwargs):
        if embedding is not None:
            embedding = F.silu(self.projection1(F.silu(self.projection0(embedding))))
        return self.diffusion_model.sample(*args, embedding=embedding, **kwargs)

class RaveDataset(Dataset):
    def __init__(self, latent_folder, latent_files):
        self.latent_folder = latent_folder
        self.latent_files = latent_files
        self.latent_data = []
        self.has_embedding = False
        self.embedding = None

        latent_size = -1

        for latent_file in self.latent_files:
            latent_path = os.path.join(self.latent_folder, latent_file)
            z = np.load(latent_path)
            if latent_path.endswith(".npz"):
                self.has_embedding = True
                z_audio = torch.from_numpy(z["z_audio"]).float().squeeze()
                if "zcr" in z:
                    self.embedding = "zcr"
                    embedding = torch.from_numpy(z["zcr"]).float().reshape(-1, 1)
                else:
                    self.embedding = "rms"
                    embedding = torch.from_numpy(z["rms"]).float().reshape(-1, 1)
                self.latent_data.append({ "z_audio": z_audio, self.embedding: embedding })
                if latent_size == -1:
                    latent_size = z_audio.shape[0]
            else:
                z = torch.from_numpy(z).float().squeeze()
                self.latent_data.append(z)
                if latent_size == -1:
                    latent_size = z.shape[0]

        self.latent_size = latent_size

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, index):
        return self.latent_data[index]

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with a new dataset.")
    parser.add_argument("--name", type=str, default=f"run_{current_date}", help="Name of your training run.")
    parser.add_argument("--latent_folder", type=str, default="./latents/", help="Path to the directory containing the latent files.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Resume training from checkpoint.")
    parser.add_argument("--save_out_path", type=str, default="./runs/", help="Path to the directory where the model checkpoints will be saved.")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Ratio for splitting the dataset into training and validation sets.")
    parser.add_argument("--max_epochs", type=int, default=25000, help="Maximum epochs to train model.")
    parser.add_argument("--scheduler_steps", type=int, default=100, help="Diffusion steps for scheduler.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument("--save_interval", type=int, default=50, help="Interval (number of epochs) at which to save the model.")
    parser.add_argument("--finetune", type=bool, default=False, help="Finetune model.")
    parser.add_argument("--wandb_project_name", type=str, default="rave-ld", help="The project name to use for WandB logging.")
    parser.add_argument("--embedding_projection_size", type=int, default=128, help="Size of the output of the projection for the conditional embedding.")
    parser.add_argument("--eval_mse_condition_every", type=int, default=0, help="The number of epochs between logging the MSE of the conditioning (ZCR or RMS) during evaluation. If set to 0, does not log.")
    parser.add_argument("--rave_model_path", type=str, help="Path to the RAVE model checkpoint (required when --eval_mse_condition_every is > 0).")
    parser.add_argument("--embedding_mask_proba", type=float, default=0.1, help="Probability of masking the embedding (when using a condition).")
    parser.add_argument("--embedding_scale", type=float, default=5.0, help="Embedding (Guidance) scale for sampling (used when --eval_mse_condition_every is > 0).")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of steps for sampling (used when --eval_mse_condition_every is > 0).")
    return parser.parse_args()

def set_seed(seed=664):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            print("The checkpoint file does not contain the required keys. Training will start from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0

    return start_epoch

def main():
    # Parse command-line arguments
    args = parse_args()
    
    if args.eval_mse_condition_every > 0 and args.rave_model_path is None:
        raise ValueError("When --eval_mse_condition_every is > 0, --rave_model_path must be provided.")

    latent_folder = args.latent_folder
    checkpoint_path = args.checkpoint_path
    save_out_path = args.save_out_path
    split_ratio = args.split_ratio
    batch_size = args.batch_size
    save_interval = args.save_interval
    embedding_mask_proba = args.embedding_mask_proba
    embedding_scale = args.embedding_scale
    num_steps = args.num_steps

    global best_loss
    global best_epoch
    best_epoch = None
    best_loss = float('inf')

    os.makedirs(args.save_out_path, exist_ok=True)

    latent_files = [f for f in os.listdir(latent_folder) if f.endswith(".npy") or f.endswith(".npz")]

    set_seed(664)

    random.shuffle(latent_files)
    split_index = int(len(latent_files) * split_ratio)
    train_latent_files = latent_files[:split_index]
    val_latent_files = latent_files[split_index:]

    train_dataset = RaveDataset(latent_folder, train_latent_files)
    val_dataset = RaveDataset(latent_folder, val_latent_files)

    # rave_dims = train_dataset.latent_size
    rave_dims = 4 # unsure why ^ didn't work

    batch_size = args.batch_size

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    diffusion_model = DiffusionModel(
        net_t=UNetV0,
        in_channels=rave_dims,
        channels=[256, 256, 256, 256, 512, 512, 512, 768, 768],
        factors=[1, 4, 4, 4, 2, 2, 2, 1, 1],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        attention_heads=12,
        attention_features=64,
        diffusion_t=VDiffusion,
        modulation_features=1024 if not train_dataset.has_embedding else args.embedding_projection_size,
        use_embedding_cfg=train_dataset.has_embedding,
        embedding_max_length=1 if train_dataset.has_embedding else None,
        embedding_features=args.embedding_projection_size if train_dataset.has_embedding else None,
        cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1] if train_dataset.has_embedding else None,
        sampler_t=VSampler,
    )
    
    model = EmbeddingProjection(
        diffusion_model=diffusion_model,
        embedding_projection_size=args.embedding_projection_size,
    ).to(device)

    print("Model Architecture:")
    print(model)
    print("\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    print(f"Number of trainable parameters: {trainable_params}\n")

    rave_model = None
    if args.rave_model_path is not None:
        rave_model = torch.jit.load(
            args.rave_model_path,
            map_location=device,
        )
        print(f"Loaded RAVE model from {args.rave_model_path}")

    embedding_func = None
    if args.eval_mse_condition_every > 0:
        if train_dataset.embedding == "zcr":
            embedding_func = lambda x: np.mean(zero_crossing_rate(x), axis=-1)
        elif train_dataset.embedding == "rms":
            embedding_func = lambda x: np.mean(rms(y=x), axis=-1)

    print("Training:", len(train_latent_files))
    print("Validation:", len(val_latent_files))

    if checkpoint_path != None:
        print(f"Resuming training from: {checkpoint_path}\n")

    if not args.finetune:
        ##### TRAIN FROM SCRATCH
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_steps, gamma=0.99)
        start_epoch = resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler)
    else:
        #### FINETUNE
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5) # Change the learning rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_steps, eta_min=1e-6) # Replace the StepLR scheduler with the CosineAnnealingLR scheduler
        start_epoch = resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler)

    accumulation_steps = args.accumulation_steps

    wandb.init(project=args.wandb_project_name, name=args.name)

    for i in range(start_epoch, args.max_epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_data_loader):
            batch_rave_tensor = batch

            if isinstance(batch_rave_tensor, dict):
                loss = model(batch_rave_tensor["z_audio"].to(device), embedding=batch_rave_tensor[train_dataset.embedding].squeeze(-1).to(device), embedding_mask_proba=embedding_mask_proba)
            else:
                loss = model(batch_rave_tensor.to(device))

            train_loss += loss.item()

            if (step + 1) % accumulation_steps == 0:
                loss = loss / accumulation_steps
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= len(train_data_loader)
        print(f"Epoch {i+1}, train loss: {train_loss}")
        wandb.log({'epoch': i+1, 'train_loss': train_loss})

        random.shuffle(train_dataset.latent_files)

        with torch.no_grad():
            model.eval()

            val_loss = 0
            mse_loss = 0
            for batch in val_data_loader:
                batch_rave_tensor = batch

                if isinstance(batch_rave_tensor, dict):
                    loss = model(batch_rave_tensor["z_audio"].to(device), embedding=batch_rave_tensor[train_dataset.embedding].squeeze(-1).to(device), embedding_mask_proba=embedding_mask_proba)
                else:
                    loss = model(batch_rave_tensor.to(device))

                val_loss += loss.item()

                if args.eval_mse_condition_every > 0 and i % args.eval_mse_condition_every == 0 and isinstance(batch_rave_tensor, dict):
                    outputs = model.sample(batch_rave_tensor["z_audio"].to(device), embedding=batch_rave_tensor[train_dataset.embedding].squeeze(-1).to(device), num_steps=num_steps, embedding_scale=embedding_scale)
                    outputs = (outputs - outputs.mean()) / outputs.std() # normalize to mean 0, std 1

                    y = rave_model.decode(outputs)
                    zcrs = embedding_func(y.cpu().numpy())
                    mse_loss += torch.nn.functional.mse_loss(torch.tensor(zcrs), batch_rave_tensor[train_dataset.embedding]).item()

            val_loss /= len(val_data_loader)
            print(f"Epoch {i+1}, validation loss: {val_loss}")
            wandb_log = {'epoch': i+1, 'val_loss': val_loss}

            if args.eval_mse_condition_every > 0 and i % args.eval_mse_condition_every == 0:
                mse_loss /= len(val_data_loader)
                print(f"Epoch {i+1}, validation MSE loss: {mse_loss}")
                wandb_log['mse_loss'] = mse_loss

            wandb.log(wandb_log)

            # Save the best model
            if val_loss < best_loss:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': i
                }
                new_checkpoint_path = f"{save_out_path}/{args.name}_best_epoch{i}_loss_{val_loss}.pt"
                torch.save(checkpoint, new_checkpoint_path)
                print(f"Saved new best model with validation loss {val_loss}")

                # If a previous best model exists, remove it
                if best_epoch is not None:
                    old_checkpoint_path = f"{save_out_path}/{args.name}_best_epoch{best_epoch}_loss_{best_loss}.pt"
                    if os.path.exists(old_checkpoint_path):
                        os.remove(old_checkpoint_path)
                best_epoch = i
                best_loss = val_loss

            # Save a checkpoint every n epochs
            if i % save_interval == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': i
                }
                torch.save(checkpoint, f"{save_out_path}/{args.name}_epoch{i}.pt")

            scheduler.step()

    wandb.finish()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
