import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import StateSpace
from prob2_learn import MPCNet
# ==========================================
# [1] 학습 코드와 동일한 PyTorch 신경망 모델 정의
# ==========================================

def main():
    print("=== [2단계] PyTorch 모델 평가 및 시뮬레이션 시작 ===")
    
    # ----------------------------------------------------
    # 1. 학습된 PyTorch 모델 불러오기
    # ----------------------------------------------------
    model_filename = 'mpc_pytorch_model.pth'
    nn_model = MPCNet()
    
    try:
        nn_model.load_state_dict(torch.load(model_filename))
        nn_model.eval() # 추론 모드 전환
        print(f"'{model_filename}' 모델 가중치를 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: '{model_filename}' 파일을 찾을 수 없습니다. 학습 코드를 먼저 실행하세요.")
        return

    # ----------------------------------------------------
    # 2. 비교 검증을 위한 Exact mp-QP 행렬 정의
    # ----------------------------------------------------
    H = np.array([[0.7578, -0.9699], [-0.9699, 1.2428]])
    F = np.array([[ 0.1111, -0.1422], [-0.0711,  0.0911],
                  [ 0.7577, -0.9699], [-0.9699,  1.2426],
                  [-0.1010,  0.1262], [ 0.0757, -0.1010]])
    G = np.array([[ 1,  0], [-1,  0], [ 0,  1], [ 0, -1]])
    W = np.array([1, 1, 1, 1])
    E = np.array([[0, 0, -1, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, -1, 0, 0], [0, 0, 0, 1, 0, 0]])

    dU = cp.Variable(2)
    p_param = cp.Parameter(6)
    cost = 0.5 * cp.quad_form(dU, H) + (F.T @ p_param).T @ dU
    prob = cp.Problem(cp.Minimize(cost), [G @ dU <= W + E @ p_param])

    # ----------------------------------------------------
    # 3. 연속시간 시스템 정의 및 이산화 (T = 2 sec)
    # ----------------------------------------------------
    A_c = np.array([[-0.01, 0], [0, -0.01]])
    B_c = np.array([[0.4, -0.5], [-0.3, 0.4]])
    C_c = np.eye(2)
    D_c = np.zeros((2, 2))
    
    sys_c = StateSpace(A_c, B_c, C_c, D_c)
    sys_d = sys_c.to_discrete(2.0)
    Ad, Bd, Cd = sys_d.A, sys_d.B, sys_d.C

    # ----------------------------------------------------
    # 4. 시뮬레이션 설정 (스텝 수 증가 및 Step Reference 적용)
    # ----------------------------------------------------
    # 시스템 시정수(tau) = 100초 이므로 충분한 수렴 확인을 위해 300스텝(600초)으로 늘림
    sim_steps = 500  
    time_axis = np.arange(sim_steps) * 2.0  # 시간 축 (sec)

    # Reference 궤적 생성 (초기에는 [0, 0], 50스텝(100초)부터 [0.63, 0.79]로 스텝 변화)
    R_ref_history = np.zeros((sim_steps, 2))
    step_time_idx = 50
    R_ref_history[step_time_idx:, 0] = 0.63
    R_ref_history[step_time_idx:, 1] = 0.79

    # 상태 및 입력 초기화
    x_exact, u_prev_exact = np.zeros(2), np.zeros(2)
    x_ml, u_prev_ml = np.zeros(2), np.zeros(2)
    
    Y_exact_history, U_exact_history = [], []
    Y_ml_history, U_ml_history = [], []

    print("폐루프(Closed-loop) 시뮬레이션 진행 중... (총 600초)")

    # ----------------------------------------------------
    # 5. 제어 루프 실행
    # ----------------------------------------------------
    for k in range(sim_steps):
        # 현재 시점의 목표치 가져오기
        r_ref = R_ref_history[k]
        
        # --- (A) Exact MPC 제어 ---
        p_exact = np.concatenate((x_exact, u_prev_exact, r_ref))
        p_param.value = p_exact
        try:
            prob.solve(solver=cp.OSQP)
            if prob.status == cp.OPTIMAL:
                dU_exact = dU.value
            else:
                dU_exact = np.zeros(2)
        except:
            dU_exact = np.zeros(2)
            
        u_exact = np.clip(u_prev_exact + dU_exact, -1.0, 1.0) 
        
        # --- (B) ML Approximation 제어 ---
        p_ml = np.concatenate((x_ml, u_prev_ml, r_ref))
        
        p_tensor = torch.FloatTensor(p_ml).unsqueeze(0)
        with torch.no_grad():
            dU_ml_tensor = nn_model(p_tensor)
            dU_ml = dU_ml_tensor.numpy()[0]
            
        u_ml = np.clip(u_prev_ml + dU_ml, -1.0, 1.0)
        
        # --- (C) 데이터 기록 및 상태 업데이트 ---
        Y_exact_history.append(Cd @ x_exact)
        Y_ml_history.append(Cd @ x_ml)
        U_exact_history.append(u_exact)
        U_ml_history.append(u_ml)
        
        x_exact = Ad @ x_exact + Bd @ u_exact
        u_prev_exact = u_exact
        
        x_ml = Ad @ x_ml + Bd @ u_ml
        u_prev_ml = u_ml

    Y_exact, Y_ml = np.array(Y_exact_history), np.array(Y_ml_history)
    U_exact, U_ml = np.array(U_exact_history), np.array(U_ml_history)

    # ----------------------------------------------------
    # 6. 시각화 (Visualization)
    # ----------------------------------------------------
    print("결과 그래프를 출력합니다.")
    plt.figure(figsize=(14, 10))

    # [Plot 1] Output (y) 비교 그래프
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, Y_exact[:, 0], 'b-', linewidth=2.5, label='Exact MPC: $y_1$')
    plt.plot(time_axis, Y_exact[:, 1], 'r-', linewidth=2.5, label='Exact MPC: $y_2$')
    plt.plot(time_axis, Y_ml[:, 0], 'c--', linewidth=2, label='PyTorch ML: $y_1$')
    plt.plot(time_axis, Y_ml[:, 1], 'm--', linewidth=2, label='PyTorch ML: $y_2$')
    
    # Reference 변화 궤적을 계단형(Step)으로 시각화
    plt.plot(time_axis, R_ref_history[:, 0], 'b:', linewidth=2, alpha=0.7, label='Ref $r_1$')
    plt.plot(time_axis, R_ref_history[:, 1], 'r:', linewidth=2, alpha=0.7, label='Ref $r_2$')

    plt.title('Output Responses (Step Tracking): Exact MPC vs PyTorch ML', fontsize=14)
    plt.ylabel('Outputs ($y_1, y_2$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # [Plot 2] Control Input (u) 비교 그래프
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, U_exact[:, 0], 'b-', linewidth=2.5, label='Exact MPC: $u_1$')
    plt.plot(time_axis, U_exact[:, 1], 'r-', linewidth=2.5, label='Exact MPC: $u_2$')
    plt.plot(time_axis, U_ml[:, 0], 'c--', linewidth=2, label='PyTorch ML: $u_1$')
    plt.plot(time_axis, U_ml[:, 1], 'm--', linewidth=2, label='PyTorch ML: $u_2$')
    
    plt.axhline(1.0, color='k', linestyle='-', alpha=0.8, label='Max Limit (+1)')
    plt.axhline(-1.0, color='k', linestyle='-', alpha=0.8, label='Min Limit (-1)')

    plt.title('Control Inputs: Exact MPC vs PyTorch ML Approximation', fontsize=14)
    plt.xlabel('Time (sec)', fontsize=12)
    plt.ylabel('Control Inputs ($u_1, u_2$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()