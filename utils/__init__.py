import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from numpy import mean


def denoise_single(diffuser, denoiser, x_t, t, cfg):
    x_hat = x_t
    denoiser.eval()
    for i in reversed(range(1, t.item())):
        ti = (torch.ones(1) * i).long().to(cfg.device)
        eps_hat = denoiser(x_hat, ti)
        alpha = diffuser.alpha[t][:, None, None]
        alpha_hat = diffuser.alpha_hat[t][:, None, None]
        x_hat = 1 / torch.sqrt(alpha) * (x_hat - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * eps_hat)
    denoiser.train()
    return x_hat


def levenshtein_dist(x, y):
    m, n = x.shape[0], y.shape[0]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def levenshtein_dist_batch(x, y):
    x, y = torch.argmax(x, dim=-1), torch.argmax(y, dim=-1)
    dist = 0
    for xi, yi in zip(x, y):
        dist += levenshtein_dist(xi.to('cpu'), yi.to('cpu'))
    return dist / x.shape[0]


def calculate_metrics(y_true, y_pred):
    accs, recalls, precisions, f1s = [], [], [], []
    for yti, ypi in zip(y_true, y_pred):
        accs.append(accuracy_score(yti, ypi))
        precisions.append(precision_score(yti, ypi, average='macro', zero_division=0))
        recalls.append(recall_score(yti, ypi, average='macro', zero_division=0))
        f1s.append(f1_score(yti, ypi, average='macro', zero_division=0))
        # aucs.append(roc_auc_score(yti, ypi, average='macro', multi_class='ovr'))
    return mean(accs), mean(recalls), mean(precisions), mean(f1s)
