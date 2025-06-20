import json
import os
import pickle as pkl
import random
import warnings
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import TraceDataset
from diffusion.Diffusion import Diffusion
from denoisers.ConditionalUnetDenoiser import ConditionalUnetDenoiser
from denoisers.ConditionalUnetMatrixDenoiser import ConditionalUnetMatrixDenoiser
from utils.initialization import initialize
from utils.pm_utils import discover_dk_process, remove_duplicates_dataset, pad_to_multiple_of_n, \
    get_process_model_petri_net_flow_matrix, get_process_model_reachability_graph_transition_multimatrix

warnings.filterwarnings("ignore")


def save_ckpt(model, opt, epoch, cfg, train_loss, test_loss, best=False):
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    torch.save(ckpt, os.path.join(cfg.summary_path, 'last.ckpt'))
    if best:
        torch.save(ckpt, os.path.join(cfg.summary_path, 'best.ckpt'))


def evaluate_batch(gt, predicted, cfg):
    for dk_trace, dk_hat_trace in zip(gt, predicted):
        dk_activities = torch.argmax(dk_trace, dim=0).cpu().numpy()
        dk_hat_activities = torch.argmax(torch.softmax(dk_hat_trace, dim=1), dim=1).cpu().numpy()
        accs = metrics.accuracy_score(dk_activities[dk_activities != cfg.pad_token],
                                      dk_hat_activities[dk_activities != cfg.pad_token])
        pres = metrics.precision_score(dk_activities[dk_activities != cfg.pad_token],
                                       dk_hat_activities[dk_activities != cfg.pad_token], average='macro',
                                       zero_division=0)
        recs = metrics.recall_score(dk_activities[dk_activities != cfg.pad_token],
                                    dk_hat_activities[dk_activities != cfg.pad_token], average='macro',
                                    zero_division=0)
        dists = wasserstein_distance(dk_activities[dk_activities != cfg.pad_token],
                                     dk_hat_activities[dk_activities != cfg.pad_token])
        return accs, pres, recs, dists


def evaluate(diffuser, denoiser, test_loader, transition_matrix, cfg, summary, epoch):
    denoiser.eval()
    total_loss = 0.0
    sequence_loss = 0.0
    matrix_loss = 0.0
    results_accumulator = {'x': [], 'y': [], 'x_hat': []}
    l = len(test_loader)
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.permute(0, 2, 1).to(cfg.device).float()
            y = y.permute(0, 2, 1).to(cfg.device).float()
            x_hat, matrix_hat, loss, seq_loss, mat_loss = \
                diffuser.sample_with_matrix(denoiser, y.shape[0], cfg.num_classes, denoiser.max_input_dim,
                                            transition_matrix.shape[-1], transition_matrix, x, y,
                                            cfg.predict_on)
            results_accumulator['x'].append(x)
            results_accumulator['y'].append(y)
            results_accumulator['x_hat'].append(x_hat.permute(0, 2, 1))
            total_loss += loss
            sequence_loss += seq_loss
            matrix_loss += mat_loss
            summary.add_scalar("test_loss", loss, global_step=epoch * l + i)
            summary.add_scalar("test_sequence_loss", seq_loss, global_step=epoch * l + i)
            summary.add_scalar("test_matrix_loss", mat_loss, global_step=epoch * l + i)

        accs, pres, recs, dists = [], [], [], []
        for x, x_hat in zip(results_accumulator['x'], results_accumulator['x_hat']):
            batch_accs, batch_pres, batch_recs, batch_dists = evaluate_batch(x, x_hat, cfg)
            accs.append(batch_accs)
            pres.append(batch_pres)
            recs.append(batch_recs)
            dists.append(batch_dists)

        accuracy = np.mean(accs)
        precision = np.mean(pres)
        recall = np.mean(recs)
        w2 = np.mean(dists)
        average_loss = total_loss / l
        average_first_loss = sequence_loss / l
        average_second_loss = matrix_loss / l

        summary.add_scalar("test_w2", w2, global_step=epoch * l)
        summary.add_scalar("test_accuracy", accuracy, global_step=epoch * l)
        summary.add_scalar("test_recall", recall, global_step=epoch * l)
        summary.add_scalar("test_precision", precision, global_step=epoch * l)
        summary.add_scalar("test_alpha", denoiser.alpha, global_step=epoch * l)
        denoiser.train()
    return average_loss, accuracy, recall, precision, w2, average_first_loss, average_second_loss, \
        denoiser.alpha


def train(diffuser, denoiser, optimizer, train_loader, test_loader, transition_matrix, cfg, summary, logger):
    test_losses, test_dist, test_acc, test_precision, test_recall = [], [], [], [], []
    train_losses, train_dist, train_acc, train_precision, train_recall = [], [], [], [], []
    train_seq_loss, train_matrix_loss, test_seq_loss, test_matrix_loss = [], [], [], []
    train_alpha, test_alpha = [], []
    l = len(train_loader)
    transition_matrix = transition_matrix.unsqueeze(0)
    best_loss = float('inf')
    denoiser.train()
    for epoch in tqdm(range(cfg.num_epochs)):
        l_matrix = 0  # how many times matrix loss was calculated because it's dropped out sometimes
        epoch_loss = 0.0
        epoch_first_loss = 0.0
        epoch_second_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.permute(0, 2, 1).to(cfg.device).float()
            y = y.permute(0, 2, 1).to(cfg.device).float()
            t = diffuser.sample_timesteps(x.shape[0]).to(cfg.device)
            drop_matrix = False
            x_t, eps = diffuser.noise_data(x, t)  # each item in batch gets different level of noise based on timestep
            if np.random.random() < cfg.conditional_dropout:
                y = None
            if np.random.random() < cfg.matrix_dropout:
                drop_matrix = True
            output, matrix_hat, loss, seq_loss, mat_loss = denoiser(x_t, t, x, transition_matrix, y, drop_matrix)
            loss.backward()
            optimizer.step()

            summary.add_scalar("train_loss", loss.item(), global_step=epoch * l + i)
            epoch_loss += loss.item()
            epoch_first_loss += seq_loss
            epoch_second_loss += mat_loss
            summary.add_scalar("train_sequence_loss", seq_loss, global_step=epoch * l + i)
            if mat_loss != 0:
                l_matrix += 1
                summary.add_scalar("train_matrix_loss", mat_loss, global_step=epoch * l + i)
        train_losses.append(epoch_loss / l)
        train_seq_loss.append(epoch_first_loss / l)
        train_matrix_loss.append(epoch_second_loss / max(l_matrix, 1))
        train_alpha.append(denoiser.alpha)

        if epoch % cfg.test_every == 0:
            logger.info("testing epoch")
            if cfg.eval_train:
                denoiser.eval()
                with (torch.no_grad()):
                    sample_index = random.choice(range(len(train_loader)))
                    for i, batch in enumerate(train_loader):
                        if i == sample_index:
                            x, y = batch
                            break
                    x = x.permute(0, 2, 1).to(cfg.device).float()
                    y = y.permute(0, 2, 1).to(cfg.device).float()
                    output, matrix_hat, loss, seq_loss, mat_loss = \
                        diffuser.sample_with_matrix(denoiser, y.shape[0], cfg.num_classes, denoiser.max_input_dim,
                                                    transition_matrix.shape[-1], transition_matrix, x, y,
                                                    cfg.predict_on)

                    accs, pres, recs, dists = [], [], [], []
                    batch_accs, batch_pres, batch_recs, batch_dists = evaluate_batch(x, output.permute(0, 2, 1), cfg)
                    accs.append(batch_accs)
                    pres.append(batch_pres)
                    recs.append(batch_recs)
                    dists.append(batch_dists)

                    accuracy = np.mean(accs)
                    precision = np.mean(pres)
                    recall = np.mean(recs)
                    w2 = np.mean(dists)
                    train_acc.append(accuracy)
                    train_recall.append(recall)
                    train_precision.append(precision)
                    train_dist.append(w2)
                    summary.add_scalar("train_w2", w2, global_step=epoch * l)
                    summary.add_scalar("train_accuracy", accuracy, global_step=epoch * l)
                    summary.add_scalar("train_recall", recall, global_step=epoch * l)
                    summary.add_scalar("train_precision", precision, global_step=epoch * l)
                    summary.add_scalar("train_alpha", denoiser.alpha, global_step=epoch * l)
                denoiser.train()

            test_epoch_loss, test_epoch_acc, test_epoch_recall, test_epoch_precision, \
                test_epoch_dist, test_epoch_seq_loss, test_epoch_mat_loss, test_alpha_clamp = \
                evaluate(diffuser, denoiser, test_loader, transition_matrix, cfg, summary, epoch)
            test_dist.append(test_epoch_dist)
            test_losses.append(test_epoch_loss)
            test_acc.append(test_epoch_acc)
            test_recall.append(test_epoch_recall)
            test_precision.append(test_epoch_precision)
            test_seq_loss.append(test_epoch_seq_loss)
            test_matrix_loss.append(test_epoch_mat_loss)
            test_alpha.append(test_alpha_clamp)
            logger.info("saving model")
            save_ckpt(denoiser, optimizer, epoch, cfg, train_losses[-1], test_losses[-1],
                      test_epoch_loss < best_loss)
            best_loss = test_epoch_loss if test_epoch_loss < best_loss else best_loss

    return (train_losses, test_losses, test_dist, test_acc, test_precision, test_recall,
            train_acc, train_recall, train_precision, train_dist, train_seq_loss,
            train_matrix_loss, test_seq_loss, test_matrix_loss, train_alpha, test_alpha)


def main():
    args, cfg, dataset, logger = initialize()
    salads_dataset = TraceDataset(dataset['target'], dataset['stochastic'])
    train_dataset, test_dataset = train_test_split(salads_dataset, train_size=cfg.train_percent, shuffle=True,
                                                   random_state=cfg.seed)
    logger.info(f"train size: {len(train_dataset)} test size: {len(test_dataset)}")
    # random initialization instead of None for compatibility, isn't used in any way if enable_matrix is false
    rg_transition_matrix = torch.randn((cfg.num_classes, 2, 2)).to(cfg.device)
    dk_process_model, dk_init_marking, dk_final_marking = discover_dk_process(train_dataset, cfg,
                                                                              preprocess=remove_duplicates_dataset)
    if cfg.enable_matrix:
        if cfg.matrix_type == "pm":
            rg_nx, rg_transition_matrix = get_process_model_petri_net_flow_matrix(dk_process_model,
                                                                                  dk_init_marking,
                                                                                  dk_final_marking)
            rg_transition_matrix = torch.tensor(rg_transition_matrix, device=cfg.device).unsqueeze(0).float()
        elif cfg.matrix_type == "rg":
            rg_nx, rg_transition_matrix = get_process_model_reachability_graph_transition_multimatrix(dk_process_model,
                                                                                                      dk_init_marking)
            rg_transition_matrix = torch.tensor(rg_transition_matrix, device=cfg.device).float()
        rg_transition_matrix = pad_to_multiple_of_n(rg_transition_matrix)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    diffuser = Diffusion(noise_steps=cfg.num_timesteps, device=cfg.device)

    if cfg.enable_matrix:
        denoiser = ConditionalUnetMatrixDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                                 max_input_dim=salads_dataset.sequence_length,
                                                 transition_dim=rg_transition_matrix.shape[-1],
                                                 gamma=cfg.gamma,
                                                 matrix_out_channels=rg_transition_matrix.shape[0],
                                                 device=cfg.device).to(cfg.device).float()
    else:
        denoiser = ConditionalUnetDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                           max_input_dim=salads_dataset.sequence_length,
                                           device=cfg.device).to(cfg.device).float()
    if cfg.parallelize:
        denoiser = nn.DataParallel(denoiser, device_ids=[0, 1])

    optimizer = AdamW(denoiser.parameters(), cfg.learning_rate)
    summary = SummaryWriter(cfg.summary_path)

    (train_losses, test_losses, test_dist, test_acc, test_precision, tests_recall, train_acc,
     train_recall, train_precision, train_dist, train_seq_loss, train_mat_loss, test_seq_loss,
     test_mat_loss, train_alpha, test_alpha) = \
        train(diffuser, denoiser, optimizer, train_loader, test_loader, rg_transition_matrix, cfg, summary, logger)

    px.line(train_losses).write_html(os.path.join(cfg.summary_path, "train_loss.html"))
    px.line(test_losses).write_html(os.path.join(cfg.summary_path, "test_loss.html"))
    px.line(test_dist).write_html(os.path.join(cfg.summary_path, "test_dist.html"))
    px.line(test_acc).write_html(os.path.join(cfg.summary_path, "test_acc.html"))
    px.line(test_precision).write_html(os.path.join(cfg.summary_path, "test_precision.html"))
    px.line(tests_recall).write_html(os.path.join(cfg.summary_path, "test_recall.html"))
    px.line(train_acc).write_html(os.path.join(cfg.summary_path, "train_acc.html"))
    px.line(train_recall).write_html(os.path.join(cfg.summary_path, "train_recall.html"))
    px.line(train_precision).write_html(os.path.join(cfg.summary_path, "train_precision.html"))
    px.line(train_dist).write_html(os.path.join(cfg.summary_path, "train_dist.html"))
    px.line(train_seq_loss).write_html(os.path.join(cfg.summary_path, "train_seq_loss.html"))
    px.line(train_mat_loss).write_html(os.path.join(cfg.summary_path, "train_mat_loss.html"))
    px.line(test_seq_loss).write_html(os.path.join(cfg.summary_path, "test_seq_loss.html"))
    px.line(test_mat_loss).write_html(os.path.join(cfg.summary_path, "test_mat_loss.html"))
    px.line(train_alpha).write_html(os.path.join(cfg.summary_path, "train_alpha.html"))
    px.line(test_alpha).write_html(os.path.join(cfg.summary_path, "test_alpha.html"))

    final_results = {
        "train":
            {
                "loss": train_losses[-1],
                "acc": train_acc[-1],
                "precision": train_precision[-1],
                "recall": train_recall[-1],
                "dist": train_dist[-1]
            },
        "test":
            {
                "loss": test_losses[-1],
                "acc": test_acc[-1],
                "precision": test_precision[-1],
                "recall": tests_recall[-1],
                "dist": test_dist[-1]
            }
    }
    with open(os.path.join(cfg.summary_path, "final_results.json"), "w") as f:
        json.dump(final_results, f)


if __name__ == "__main__":
    main()
