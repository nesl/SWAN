import torch

def compare_state_dicts(dict1, dict2, epsilon=1e-8):
    """
    Compares two PyTorch state dicts and returns keys where weights differ.
    """
    # Find common keys to avoid KeyErrors
    common_keys = set(dict1.keys()) & set(dict2.keys())
    differences = {}

    for key in common_keys:
        val1 = dict1[key]
        val2 = dict2[key]
        if not torch.is_floating_point(val1):
            val1 = val1.float()
        if not torch.is_floating_point(val2):
            val2 = val2.float()
        # Check if the weights are close within tolerance epsilon
        if not torch.allclose(val1, val2, atol=epsilon):
            # Calculate the absolute difference for context
            diff_norm = torch.norm(val1 - val2).item()
            differences[key] = diff_norm
            
    return differences

# --- Example Usage ---

# 1. Setup dummy state dicts
model_a = torch.load('/workspace/mmdetection3d/work_dirs/cmt_train_all_corruptions/epoch_8.pth')['state_dict']
model_b = torch.load('/workspace/mmdetection3d/work_dirs/ADMN_20_early_exit/epoch_3.pth')['state_dict']


# 2. Run comparison
epsilon_val = 1e-5
diffs = compare_state_dicts(model_a, model_b, epsilon=epsilon_val)

# 3. Report findings
if not diffs:
    print(f"✅ All common weights are identical within epsilon {epsilon_val}")
else:
    print(f"❌ Found {len(diffs)} differing layers:")
    for key, norm in diffs.items():
        print(f" - {key}: Frobenius norm of diff = {norm:.6f}")

# class NeuralSort (torch.nn.Module):
#     def __init__(self, tau=1.0, hard=False):
#         super(NeuralSort, self).__init__()
#         self.hard = hard
#         self.tau = tau

#     def forward(self, scores):
#         """
#         scores: elements to be sorted. Typical shape: batch_size x n x 1
#         """
#         scores = scores.unsqueeze(-1)
#         bsize = scores.size()[0]
#         dim = scores.size()[1]
#         one = torch.cuda.FloatTensor(dim, 1).fill_(1)

#         A_scores = torch.abs(scores - scores.permute(0, 2, 1))
#         B = torch.matmul(A_scores, torch.matmul(
#             one, torch.transpose(one, 0, 1)))
#         scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
#                    ).type(torch.cuda.FloatTensor)
#         C = torch.matmul(scores, scaling.unsqueeze(0))

#         P_max = (C-B).permute(0, 2, 1)
#         sm = torch.nn.Softmax(-1)
#         P_hat = sm(P_max / self.tau)

#         if self.hard:
#             P = torch.zeros_like(P_hat, device='cuda')
#             b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
#                 dim0=1, dim1=0).flatten().type(torch.cuda.LongTensor)
#             r_idx = torch.arange(dim).repeat(
#                 [bsize, 1]).flatten().type(torch.cuda.LongTensor)
#             c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
#             brc_idx = torch.stack((b_idx, r_idx, c_idx))

#             P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
#             P_hat = (P-P_hat).detach() + P_hat
#         return P_hat
    
# test = NeuralSort()
# rand = torch.randn(1,5).cuda()
# res =test(rand)
# import pdb; pdb.set_trace()