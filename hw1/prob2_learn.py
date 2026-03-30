import numpy as np
import cvxpy as cp
from scipy.signal import StateSpace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
    print("=== [1단계] 데이터 생성 및 PyTorch 신경망 학습 시작 ===")
    
    # ----------------------------------------------------
    # 1. 과제에 주어진 mp-QP 행렬 정의
    # ----------------------------------------------------
    H = np.array([[0.7578, -0.9699], [-0.9699, 1.2428]])
    F = np.array([[ 0.1111, -0.1422], [-0.0711,  0.0911],
                  [ 0.7577, -0.9699], [-0.9699,  1.2426],
                  [-0.1010,  0.1262], [ 0.0757, -0.1010]])
    G = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    W = np.array([1, 1, 1, 1])
    E = np.array([[0, 0, -1, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, -1, 0, 0], [0, 0, 0, 1, 0, 0]])

    # ----------------------------------------------------
    # 2. 오프라인 데이터 생성 (Exact QP 풀이)
    # ----------------------------------------------------
    num_samples = 10000
    P_train = np.random.uniform(low=-2.0, high=2.0, size=(num_samples, 6))
    U_train = np.zeros((num_samples, 2))

    dU = cp.Variable(2)
    p_param = cp.Parameter(6)
    cost = 0.5 * cp.quad_form(dU, H) + (F.T @ p_param).T @ dU
    prob = cp.Problem(cp.Minimize(cost), [G @ dU <= W + E @ p_param])

    print(f"{num_samples}개의 샘플에 대해 Exact MPC QP 푸는 중...")
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
    print(f"유효한 데이터셋 생성 완료: {len(valid_idx)} 샘플\n")

    # ----------------------------------------------------
    # 3. PyTorch 데이터셋 및 DataLoader 준비
    # ----------------------------------------------------
    # Numpy 배열을 PyTorch Tensor로 변환
    X_tensor = torch.FloatTensor(P_valid)
    Y_tensor = torch.FloatTensor(U_valid)

    # TensorDataset과 DataLoader 생성 (배치 학습용)
    batch_size = 128
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ----------------------------------------------------
    # 4. PyTorch 모델 초기화 및 학습 설정
    # ----------------------------------------------------
    model = MPCNet()
    criterion = nn.MSELoss() # 손실 함수: Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 최적화: Adam (학습률 0.001)

    epochs = 200 # 학습 반복 횟수
    print(f"PyTorch 신경망(MLP) 훈련 시작... (총 {epochs} Epochs)")

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_Y in dataloader:
            # 1. 기울기 초기화
            optimizer.zero_grad()
            
            # 2. 순전파 (Forward pass)
            predictions = model(batch_X)
            
            # 3. 손실 계산 (Loss computation)
            loss = criterion(predictions, batch_Y)
            
            # 4. 역전파 및 가중치 업데이트 (Backward pass & Optimize)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        # 평균 에포크 손실 계산
        epoch_loss /= len(dataloader.dataset)
        
        # 20 에포크마다 진행 상황 출력
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}], Loss(MSE): {epoch_loss:.6f}")

    print("\n학습 완료!")

    # ----------------------------------------------------
    # 5. 모델 가중치(State Dict) 저장
    # ----------------------------------------------------
    model_filename = 'mpc_pytorch_model.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"PyTorch 모델 가중치가 '{model_filename}'로 저장되었습니다.")

if __name__ == "__main__":
    main()