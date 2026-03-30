import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from qpsolvers import solve_qp

class MPCNet(nn.Module):
    def __init__(self):
        super(MPCNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4) 
        )

    def forward(self, x):
        return self.network(x)

def main():

    H = np.array([[0.7578, -0.9699], [-0.9699, 1.2428]])
    F = np.array([[ 0.08889, -0.11102],
                  [-0.06659,  0.08891],
                  [ 0.7577, -0.9699],
                  [-0.9699,  1.2426],
                  [-0.1010,  0.1262],
                  [ 0.0757, -0.1010]])
    G = np.array([[ 1,  0], [-1,  0], [ 0,  1], [ 0, -1]])
    W = np.array([1, 1, 1, 1])
    E = np.array([
        [0, 0, -1,  0, 0, 0],
        [0, 0,  1,  0, 0, 0],
        [0, 0,  0, -1, 0, 0],
        [0, 0,  0,  1, 0, 0]
    ])

    num_samples = 10000
    P_train = np.random.uniform(low=-2.0, high=2.0, size=(num_samples, 6))
    
    valid_P = []
    valid_A = [] 

    print(f"Solving MPC via Active Set Solver...")
    
    P_qp = H.astype(np.float64)
    G_qp = G.astype(np.float64)
    
    for i in range(num_samples):
        p_val = P_train[i, :]
        
        q_qp = (F.T @ p_val).astype(np.float64)
        h_qp = (W + E @ p_val).astype(np.float64)
        
        try:
            du_opt = solve_qp(P_qp, q_qp, G_qp, h_qp, solver='quadprog')
            
            if du_opt is not None:
                residual = G_qp @ du_opt - h_qp
                is_active = (np.abs(residual) < 1e-4).astype(np.float32)
                
                valid_P.append(p_val)
                valid_A.append(is_active) 
        except ValueError:
            pass 

    valid_P = np.array(valid_P)
    valid_A = np.array(valid_A)
    print("Data Generation Done.\n")

    X_tensor = torch.FloatTensor(valid_P)
    Y_tensor = torch.FloatTensor(valid_A) 
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = MPCNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion = nn.BCEWithLogitsLoss()

    epochs = 200
    print("Start Learning...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_preds = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
            predicted = (torch.sigmoid(logits) > 0.5).float()
            correct_preds += (predicted == batch_y).all(dim=1).sum().item()
            
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(dataset)
            accuracy = (correct_preds / len(dataset)) * 100
            print(f"Epoch [{epoch+1:3d}/{epochs}], Loss(BCE): {avg_loss:.4f}, Exact Match Accuracy: {accuracy:.2f}%")

    print("\nLearning Done.")

    model_filename = 'prob3_model.pth'
    print("Saving Classification Model...")
    torch.save(model.state_dict(), model_filename)
    print("Saving Model Done.")

if __name__ == "__main__":
    main()