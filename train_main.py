import copy
import pickle
import time
from datetime import datetime

import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torch_geometric.loader import DataLoader

from scripts.arguments import get_train_args_parser
from scripts.conditioning import Conditioner
from scripts.dataset import FragmentLibrary, TrainCollator, TrainDataset
from scripts.model import DeepBioisostere
from scripts.train import LR_Scheduler, Trainer
from scripts.utils import Logger, set_cuda_visible_devices, set_seed, train_path_setting

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def main(args):
    # Read arguments
    args = train_path_setting(args)
    logger = Logger(name="Training Logger", save_path=f"{args.save_dir}/training.log")
    logger("\n===== Training Arguments =====")
    logger.log_args(args)

    # Initial settings
    torch.set_num_threads(args.num_cores)
    set_seed(args.seed)

    # Model settings
    model = DeepBioisostere(args)
    model.init_params()
    logger(
        f"number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    if args.use_cuda:
        logger("GPU machine was found.")
        model.cuda()
    device = model.device
    logger(f"device: {device}")

    # Model training hyperparameters
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = LR_Scheduler(
        optimizer,
        factor=args.lr_reduce_factor,
        patience=args.patience,
        threshold=args.threshold,
        min_lr=args.min_lr,
    )

    # Dataset and loader settings
    logger("\n===== Settings for Training=====")

    # Conditioning module
    if args.conditioning:
        conditioning_module = Conditioner(
            phase="train",
            properties=args.properties,
            use_delta=args.use_delta,
        )
    else:
        conditioning_module = None

    # Training and validation fragment libraries
    logger("Loading the fragment libraries...")
    insertion_frag_lib = FragmentLibrary.get_insertion_frag_library(args.data_dir, new_frag_type="all", with_maskings=True)

    # Training and validation dataset
    logger("Loading the pair-wise training data...")
    train_dataset, val_dataset = TrainDataset.get_datasets(
        data_path=args.data_path,
        conditioner=conditioning_module,
        modes=("train", "val"),
    )

    # Training and validation samplers
    logger("Setting the data samplers and loaders...")
    if args.num_batch_each_epoch:
        train_num_samples = args.batch_size * args.num_batch_each_epoch
        val_num_samples = args.batch_size * args.num_batch_each_epoch
    else:
        train_num_samples = len(train_dataset)
        val_num_samples = len(val_dataset)
    args.weighted_sampler = False
    if args.weighted_sampler:
        train_dataset_weights = train_dataset.get_data_weights(args.alpha2)
        val_dataset_weights = val_dataset.get_data_weights(args.alpha2)
        train_sampler = WeightedRandomSampler(
            train_dataset_weights, replacement=False, num_samples=train_num_samples
        )
        val_sampler = WeightedRandomSampler(
            val_dataset_weights, replacement=False, num_samples=val_num_samples
        )
    else:
        train_sampler = RandomSampler(
            train_dataset, replacement=False, num_samples=train_num_samples
        )
        val_sampler = RandomSampler(
            val_dataset, replacement=False, num_samples=val_num_samples
        )

    # Training and validation dataloaders and collaters
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cores,
        sampler=train_sampler,
        pin_memory=True,
    )
    train_data_loader.collate_fn = TrainCollator(
        fragment_library=insertion_frag_lib,
        num_neg_sample=args.num_neg_sample,
        mode="train",
        alpha1=args.alpha1,
        use_conditioning=args.conditioning,
        properties=args.properties,
        follow_batch=["x_n"],
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cores,
        sampler=val_sampler,
        pin_memory=True,
    )
    val_data_loader.collate_fn = TrainCollator(
        fragment_library=insertion_frag_lib,
        num_neg_sample=args.num_neg_sample,
        mode="val",
        alpha1=args.alpha1,
        use_conditioning=args.conditioning,
        properties=args.properties,
        follow_batch=["x_n"],
    )

    # Tensorboard profiler setting
    if args.profiling:
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f'./profile_log/{args.save_dir.split("/")[-1]}'
            ),
            record_shapes=True,
            with_stack=False,
        )
    else:
        profiler = None

    # Traniner setting
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        num_neg_sample=args.num_neg_sample,
        batch_size=args.batch_size,
        device=device,
        profiler=profiler,
    )
    trainer.set_data_loaders(
        train_data_loader,
        val_data_loader,
    )
    logger("Done.")

    # Starting model training
    logger("\n===== Training Started =====")
    since = time.time()
    now = datetime.now()
    train_start = now.strftime("%Y. %m. %d (%a) %H:%M:%S")
    logger(f"Training started at: {train_start}")
    logger("PPO: Avg Positive Position Probability (->1)")
    logger("NPO: Avg Negative Position Probability (->0)")
    logger("PFR: Avg Positive Fragment Probability (->1)")
    logger("NFR: Avg Negative Fragment probability (->0)")
    logger("ATT: Avg Pair-wise Attachment Loss (->0)")
    logger("TOT: Avg Sum of FIVE Losses (->0)")
    logger(
        "\nEP   | "
        + "TR_PPO | "
        + "TR_NPO | "
        + "TR_PFR | "
        + "TR_NFR | "
        + "TR_ATT | "
        + "TR_TOT | "
        + "VA_PPO | "
        + "VA_NPO | "
        + "VA_PFR | "
        + "VA_NFR | "
        + "VA_ATT | "
        + "VA_TOT | "
        + "EP_TIME | "
        + "LR(*1e3) | "
        + "BEST_EP"
    )

    # Logging the training process
    phases = ["train", "val"]
    best_loss = 10000
    best_epoch = 0
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, args.max_epoch + 1):
        epoch_st = time.time()
        for phase in phases:
            # Train phase
            if phase == "train":
                model.train()
                optimizer.zero_grad()
                (
                    train_pposloss,
                    train_nposloss,
                    train_pfloss,
                    train_nfloss,
                    train_attloss,
                    train_pposprob,
                    train_nposprob,
                    train_pfprob,
                    train_nfprob,
                ) = trainer.train()
                train_tot = (
                    train_pposloss
                    + train_nposloss
                    + train_pfloss
                    + train_nfloss
                    + train_attloss
                )
                train_loss_history.append(train_tot)

            # Validate phase
            elif phase == "val":
                model.eval()
                (
                    val_pposloss,
                    val_nposloss,
                    val_pfloss,
                    val_nfloss,
                    val_attloss,
                    val_pposprob,
                    val_nposprob,
                    val_pfprob,
                    val_nfprob,
                ) = trainer.validate()
                val_tot = (
                    val_pposloss + val_nposloss + val_pfloss + val_nfloss + val_attloss
                )
                val_loss_history.append(val_tot)
                _reached_optimum = scheduler.step(val_tot)

        # Save the model every 5 epoches
        if epoch % 5 == 0:
            model.save_model(model.state_dict(), args.save_dir, f"{epoch}epoch_model")

        # Check a current model is the best one
        if best_loss > val_tot:
            best_loss = val_tot
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            model.save_model(best_model, args.save_dir)

        # Logging the probabilities
        logger(
            f"{str(epoch).ljust(4)} | "
            + f"{train_pposprob:.4f} | "
            + f"{train_nposprob:.4f} | "
            + f"{train_pfprob:.4f} | "
            + f"{train_nfprob:.4f} | "
            + "       | "
            + f"{train_tot:.4f} | "
            + f"{val_pposprob:.4f} | "
            + f"{val_nposprob:.4f} | "
            + f"{val_pfprob:.4f} | "
            + f"{val_nfprob:.4f} | "
            + "       | "
            + f"{val_tot:.4f} | "
            + f"{str(time.time() - epoch_st)[:7]} | "
            + f'{optimizer.param_groups[0]["lr"]*1000:.5f}  | '
            + f"{best_epoch}"
        )
        # Logging the losses
        if args.print_loss:
            logger(
                "loss | "
                + f"{train_pposloss:.4f} | "
                + f"{train_nposloss:.4f} | "
                + f"{train_pfloss:.4f} | "
                + f"{train_nfloss:.4f} | "
                + f"{train_attloss:.4f} | "
                + f"{train_tot:.4f} | "
                + f"{val_pposloss:.4f} | "
                + f"{val_nposloss:.4f} | "
                + f"{val_pfloss:.4f} | "
                + f"{val_nfloss:.4f} | "
                + f"{val_attloss:.4f} | "
                + f"{val_tot:.4f} "
            )

        # Determining whether training should be terminated or not
        if _reached_optimum and args.lr_scheduler_can_terminate:
            logger(
                f"In epoch {epoch}, lr already reached min_lr and val loss did not decreased in {args.patience} epochs."
            )
            break

    # Finishing log
    now = datetime.now()
    finished_at = now.strftime("%Y. %m. %d (%a) %H:%M:%S")
    time_elapsed = int(time.time() - since)
    logger("\n===== Training Finished =====")
    logger(f"finished at : {finished_at}")
    logger(
        "time passed: [%dh:%dm:%ds]"
        % (time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60)
    )
    logger(f"\nBest epoch: {best_epoch}")
    logger(f"Best val loss: {best_loss:3f}")
    logger(f'Decayed lr: {optimizer.param_groups[0]["lr"]}')

    # Save the loss history
    with open(f"{args.save_dir}/loss_history.pkl", "wb") as fw:
        pickle.dump({"train": train_loss_history, "val": val_loss_history}, fw)
    return True


# main operation:
if __name__ == "__main__":
    args = get_train_args_parser()
    main(args)
