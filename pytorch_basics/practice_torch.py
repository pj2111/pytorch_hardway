# build an RNN model
# save the model without training
# work on training the model
# then save the model with the optimizer states

import torch 
from torch import nn

input_dim = 56  # 56 features 
output_dim = 1  # 1 output features
hidden_dim = 128  # 128 hidden features
epoch = 10


class RNNModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(RNNModel, self).__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.act1 = nn.ReLU() 
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.smax = nn.Softmax(dim=1)
        # self.init_weights()
 
    def forward(self, data):
        out = self.lin1(data)
        out = self.act1(out)
        out = self.lin2(out)
        return out
 
    def init_weights(self):
        torch.nn.init.uniform_(self.lin1.weight)
        torch.nn.init.uniform_(self.lin2.weight)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t_ref = torch.arange(0, 30, 1, dtype=torch.int32, device=device)
t_series = torch.cos(t_ref).view(-1, 1)

embed = nn.Embedding(30, 56, device=device)
ref_embed = embed(t_ref)

# print(ref_embed.shape)
# print(t_series)

rnn = RNNModel(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.MSELoss()
optimiser = torch.optim.SGD(params=rnn.parameters(), lr=0.01)

tot_loss = 0

for ep in range(epoch):
    # forward pass into the model
    optimiser.zero_grad()
    pred1 = rnn(ref_embed)
    loss = criterion(pred1, t_series)
    # do backward
    loss.backward(retain_graph=True)
    optimiser.step()
    tot_loss += loss.item()

    print(f"Epoch : {ep} and the loss: {tot_loss: 3f}")

print("Doing model evaluation")


def test_num():
    return embed(torch.randint(1, 30, dtype=torch.int32))


t_eval = torch.arange(0, 5, 1, dtype=torch.int32, device=device)
eval_embed = embed(t_eval)

for indx, i in enumerate(eval_embed):
    print(f"Predicted: {rnn(i)} actual: {t_series[indx]}")

checpoint = {
    "epoch": epoch,
    "model": rnn.state_dict(),
    "optimiser": optimiser.state_dict(),
}

test_model = 'model_only.pt'
test_full = 'model_full.pt'

torch.save(checpoint, test_full)
torch.save(rnn.state_dict(), test_model)
