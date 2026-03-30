import numpy as np
import cvxpy as cp
from scipy.signal import StateSpace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MPCNet(nn.Module):
    def __init__(self):
        super(MPCNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

def main():
    
    H = np.array([[0.7578, -0.9699], [-0.9699, 1.2428]])
    F = np.array([[ 0.08889, -0.11102], [-0.06659,  0.08891],
                  [ 0.7577, -0.9699], [-0.9699,  1.2426],
                  [-0.1010,  0.1262], [ 0.0757, -0.1010]])
    G = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    W = np.array([1, 1, 1, 1])
    E = np.array([[0, 0, -1, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, -1, 0, 0], [0, 0, 0, 1, 0, 0]])

    num_samples = 10000
    P_train = np.random.uniform(low=-2.0, high=2.0, size=(num_samples, 6))
    U_train = np.zeros((num_samples, 2))

    dU = cp.Variable(2)
    p_param = cp.Parameter(6)
    cost = 0.5 * cp.quad_form(dU, H) + (F.T @ p_param).T @ dU
    prob = cp.Problem(cp.Minimize(cost), [G @ dU <= W + E @ p_param])

    print("Solving MPC...")
    valid_idx = []
    for i in range(num_samples):
        p_param.value = P_train[i, :]
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            if prob.status == cp.OPTIMAL:
                U_train[i, :] = dU.value
                valid_idx.append(i)
        except:
            pass

    P_valid = P_train[valid_idx]
    U_valid = U_train[valid_idx]
    print(f"Solving MPC Done.\n")

    X_tensor = torch.FloatTensor(P_valid)
    Y_tensor = torch.FloatTensor(U_valid)

    batch_size = 128
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MPCNet()
    criterion = nn.MSELoss() # Vanilla MLP
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    epochs = 200 
    print("Start Learning...\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_Y in dataloader:

            optimizer.zero_grad()
            
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_Y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}], Loss(MSE): {epoch_loss:.6f}")

    print("\nLearning Done.")


    model_filename = 'prob2_model.pth'
    print("Saving Model...")
    torch.save(model.state_dict(), model_filename)
    print("Saving Model Done.")

if __name__ == "__main__":
    main()