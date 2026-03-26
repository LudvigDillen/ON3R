import torch
import torch.nn as nn
from omegaconf import OmegaConf


def weight_loss(log_similarity, weights, gamma=0.0, remove_query_sigmas_from_loss=False):
    b, m, n = log_similarity.shape
    m -= 1  # Because of the row containing the negative samples (sigmas)
    n -= 1  # Because of the column containing the negative samples (sigmas)

    loss_sc = log_similarity * weights  # Remove the parts of the log assignment matrix not used

    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)  # clamp to not divide by zero
    if remove_query_sigmas_from_loss:
        num_neg0 = torch.zeros_like(num_neg1)
    else:
        num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)  # clamp to not divide by zero
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)  # clamp to not divide by zero

    # Here, we do 1/|M|*sum(-log(P))
    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)


    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)
    if remove_query_sigmas_from_loss:
        nll_neg0 = torch.zeros_like(nll_neg1)
    else:
        nll_neg0 = -loss_sc[:, :m, -1].sum(-1)

    # Here, we combine the two negative parts of the loss, but slightly differently from the paper
    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)
    num_neg = (num_neg0 + num_neg1) / (2.0 - remove_query_sigmas_from_loss)
    return nll_pos, nll_neg, num_pos, num_neg


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf, remove_query_sigmas_from_loss=False, logg_sigmas=False):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss
        self.remove_query_sigmas_from_loss = remove_query_sigmas_from_loss
        self.logg_sigmas = logg_sigmas

    def forward(self, log_similarity, data, weights=None):
        if isinstance(log_similarity, dict):  # this is how default lightglue inputs it
            log_similarity = log_similarity["log_assignment"]
        if weights is None:
            weights = self.loss_fn(log_similarity, data)
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_similarity, weights, gamma=self.conf.gamma_f,
            remove_query_sigmas_from_loss=self.remove_query_sigmas_from_loss
        )
        nll = self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg

        # Logging sigmas
        if self.logg_sigmas is False:
            return nll, weights, {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            }

        query_sigmas = (1 - torch.exp(log_similarity[:, -1, :-1])).detach()
        ref_sigmas = (1 - torch.exp(log_similarity[:, :-1, -1])).detach()

        sigma_mean_query = query_sigmas.mean(dim=1)
        sigma_std_query = query_sigmas.std(dim=1)
        sigma_percentile25_query = torch.quantile(query_sigmas, 0.25, dim=1)
        sigma_percentile75_query = torch.quantile(query_sigmas, 0.75, dim=1)

        sigma_mean_ref = ref_sigmas.mean(dim=1)
        sigma_std_ref = ref_sigmas.std(dim=1)
        sigma_percentile25_ref = torch.quantile(ref_sigmas, 0.25, dim=1)
        sigma_percentile75_ref = torch.quantile(ref_sigmas, 0.75, dim=1)

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
                "sigma_mean_query": sigma_mean_query,
                "sigma_std_query": sigma_std_query,
                "sigma_percentile25_query": sigma_percentile25_query,
                "sigma_percentile75_query": sigma_percentile75_query,
                "sigma_mean_ref": sigma_mean_ref,
                "sigma_std_ref": sigma_std_ref,
                "sigma_percentile25_ref": sigma_percentile25_ref,
                "sigma_percentile75_ref": sigma_percentile75_ref,
            },
        )

    def nll_loss(self, log_similarity, data):
        """
        Weight is the mask that decides which elements to use in the loss.
        The positive part is the log(P}) part of the loss.
        The negative part is the log(1-sigma) part of the loss.
        """
        m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        # postive, neg0, and neg1 are all binary masks
        positive = data["gt_assignment"].float()
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()
        # IGNORE_FEATURE = -2  (Do not use in loss) (I think, can hold for both positive and nega.)
        # UNMATCHED_FEATURE = -1  (Use is loss for the log(1-sigma) part)
        weights = torch.zeros_like(log_similarity)
        weights[:, :m, :n] = positive
        if self.remove_query_sigmas_from_loss is False:
            weights[:, :m, -1] = neg0
        weights[:, -1, :m] = neg1
        return weights
