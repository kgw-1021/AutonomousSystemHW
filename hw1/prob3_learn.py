import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from qpsolvers import solve_qp

# ==========================================
# [1] PyTorch 신경망 모델 정의
# ==========================================
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
    print("=== [1단계] 보정된 mp-QP 행렬 + qpOASES 데이터 생성 및 학습 ===")
    
    # ----------------------------------------------------
    # 1. 완벽하게 보정된 mp-QP 행렬 정의
    # ----------------------------------------------------
    H = np.array([[0.7578, -0.9699], [-0.9699, 1.2428]])
    
    # 완벽한 기울기 균형(Gradient Balance)이 적용된 F 행렬
    F = np.array([[ 0.08889, -0.11102],
                  [-0.06659,  0.08891],
                  [ 0.7577, -0.9699],
                  [-0.9699,  1.2426],
                  [-0.1010,  0.1262],
                  [ 0.0757, -0.1010]])
    
    G = np.array([[ 1,  0], [-1,  0], [ 0,  1], [ 0, -1]])
    W = np.array([1, 1, 1, 1])
    
    # 제약조건 오류가 수정된 E 행렬
    E = np.array([
        [0, 0, -1,  0, 0, 0],
        [0, 0,  1,  0, 0, 0],
        [0, 0,  0, -1, 0, 0],
        [0, 0,  0,  1, 0, 0]
    ])

    # ----------------------------------------------------
    # 2. 오프라인 데이터 생성 (qpOASES 사용)
    # ----------------------------------------------------
    num_samples = 10000
    P_train = np.random.uniform(low=-2.0, high=2.0, size=(num_samples, 6))
    
    valid_P = []
    valid_U = []

    print(f"{num_samples}개의 샘플에 대해 qpOASES 솔버로 최적해를 계산합니다...")
    
    # qpOASES 구동을 위해 행렬들을 미리 float64로 변환
    P_qp = H.astype(np.float64)
    G_qp = G.astype(np.float64)
    
    for i in range(num_samples):
        p_val = P_train[i, :]
        
        # 1차항(q)과 부등식 우변(h) 계산 및 float64 캐스팅
        q_qp = (F.T @ p_val).astype(np.float64)
        h_qp = (W + E @ p_val).astype(np.float64)
        
        try:
            # Active Set 알고리즘 기반 qpOASES 솔버 호출
            du_opt = solve_qp(P_qp, q_qp, G_qp, h_qp, solver='quadprog')
            
            if du_opt is not None:
                valid_P.append(p_val)
                valid_U.append(du_opt)
        except ValueError:
            pass # Infeasible 케이스 무시

    valid_P = np.array(valid_P)
    valid_U = np.array(valid_U)
    print(f"유효한 데이터셋 생성 완료: {len(valid_P)} 샘플\n")

    # ----------------------------------------------------
    # 3. PyTorch 모델 학습
    # ----------------------------------------------------
    X_tensor = torch.FloatTensor(valid_P)
    Y_tensor = torch.FloatTensor(valid_U)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = MPCNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 150
    print(f"PyTorch 신경망 훈련 시작... (총 {epochs} Epochs)")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}], Loss(MSE): {epoch_loss/len(dataset):.6f}")

    # 4. 모델 저장
    model_filename = 'prob3_model.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"\n학습 완료! 모델 가중치가 '{model_filename}'로 저장되었습니다.")

if __name__ == "__main__":
    main()