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
    model_filename = 'prob2_model.pth'
    nn_model = MPCNet()
    
    try:
        nn_model.load_state_dict(torch.load(model_filename))
        nn_model.eval() 
        print(f"'{model_filename}' Model loaded.")
    except FileNotFoundError:
        print(f"Error: '{model_filename}' cannot be found.")
        return

    H = np.array([[0.7578, -0.9699], [-0.9699, 1.2428]])
    F = np.array([[ 0.08889, -0.11102], [-0.06659,  0.08891],
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

    A_c = np.array([[-0.01, 0], [0, -0.01]])
    B_c = np.array([[0.4, -0.5], [-0.3, 0.4]])
    C_c = np.eye(2)
    D_c = np.zeros((2, 2))
    
    sys_c = StateSpace(A_c, B_c, C_c, D_c)
    sys_d = sys_c.to_discrete(2.0)
    Ad, Bd, Cd = sys_d.A, sys_d.B, sys_d.C

    sim_steps = 500  
    time_axis = np.arange(sim_steps) * 2.0 

    R_ref_history = np.zeros((sim_steps, 2))
    step_time_idx = 50
    R_ref_history[step_time_idx:, 0] = 0.63
    R_ref_history[step_time_idx:, 1] = 0.79

    # 상태 및 입력 초기화
    x_exact, u_prev_exact = np.zeros(2), np.zeros(2)
    x_ml, u_prev_ml = np.zeros(2), np.zeros(2)
    
    Y_exact_history, U_exact_history = [], []
    Y_ml_history, U_ml_history = [], []
    dU_exact_history, dU_ml_history = [], []
    print("Simulation starts...")

    for k in range(sim_steps):
        r_ref = R_ref_history[k]
        
        # ---  Exact MPC  ---
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
        
        # --- Supervised Learning ---
        p_ml = np.concatenate((x_ml, u_prev_ml, r_ref))
        
        p_tensor = torch.FloatTensor(p_ml).unsqueeze(0)
        with torch.no_grad():
            dU_ml_tensor = nn_model(p_tensor)
            dU_ml = dU_ml_tensor.numpy()[0]
            
        u_ml = np.clip(u_prev_ml + dU_ml, -1.0, 1.0)
        
        Y_exact_history.append(Cd @ x_exact)
        Y_ml_history.append(Cd @ x_ml)
        U_exact_history.append(u_exact)
        U_ml_history.append(u_ml)
        dU_exact_history.append(dU_exact)
        dU_ml_history.append(dU_ml)
        
        x_exact = Ad @ x_exact + Bd @ u_exact
        u_prev_exact = u_exact
        
        x_ml = Ad @ x_ml + Bd @ u_ml
        u_prev_ml = u_ml

    Y_exact, Y_ml = np.array(Y_exact_history), np.array(Y_ml_history)
    U_exact, U_ml = np.array(U_exact_history), np.array(U_ml_history)
    dU_exact_arr, dU_ml_arr = np.array(dU_exact_history), np.array(dU_ml_history)

    mse_dU1 = np.mean((dU_exact_arr[:, 0] - dU_ml_arr[:, 0])**2)
    mse_dU2 = np.mean((dU_exact_arr[:, 1] - dU_ml_arr[:, 1])**2)
    print(f"\nResults Summary:")
    print(f"-> Predicted Parameter (dU1) MSE: {mse_dU1:.6f}")
    print(f"-> Predicted Parameter (dU2) MSE: {mse_dU2:.6f}")

    print("Result plotting...")
    plt.figure(figsize=(14, 12))

    # [Plot 1] Output (y) 비교 그래프
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, Y_exact[:, 0], 'b-', linewidth=2.5, label='Exact MPC: $y_1$')
    plt.plot(time_axis, Y_exact[:, 1], 'r-', linewidth=2.5, label='Exact MPC: $y_2$')
    plt.plot(time_axis, Y_ml[:, 0], 'c--', linewidth=2, label='Supervised Learning: $y_1$')
    plt.plot(time_axis, Y_ml[:, 1], 'm--', linewidth=2, label='Supervised Learning: $y_2$')

    plt.plot(time_axis, R_ref_history[:, 0], 'b:', linewidth=2, alpha=0.7, label='Ref $r_1$')
    plt.plot(time_axis, R_ref_history[:, 1], 'r:', linewidth=2, alpha=0.7, label='Ref $r_2$')

    plt.title('Output Responses (Step Tracking): Exact MPC vs Supervised Learning', fontsize=14)
    plt.ylabel('Outputs ($y_1, y_2$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # [Plot 2] Control Input (u) 비교 그래프
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, U_exact[:, 0], 'b-', linewidth=2.5, label='Exact MPC: $u_1$')
    plt.plot(time_axis, U_exact[:, 1], 'r-', linewidth=2.5, label='Exact MPC: $u_2$')
    plt.plot(time_axis, U_ml[:, 0], 'c--', linewidth=2, label='Supervised Learning: $u_1$')
    plt.plot(time_axis, U_ml[:, 1], 'm--', linewidth=2, label='Supervised Learning: $u_2$')

    plt.axhline(1.0, color='k', linestyle='-', alpha=0.8, label='Max Limit (+1)')
    plt.axhline(-1.0, color='k', linestyle='-', alpha=0.8, label='Min Limit (-1)')

    plt.title('Control Inputs', fontsize=14)
    plt.xlabel('Time (sec)', fontsize=12)
    plt.ylabel('Control Inputs ($u_1, u_2$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # [Plot 3] Direct Output Parameter Comparison (Delta u)
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, dU_exact_arr[:, 0], 'b-', linewidth=2.5, label='Exact MPC: $Delta u_1$')
    plt.plot(time_axis, dU_exact_arr[:, 1], 'r-', linewidth=2.5, label='Exact MPC: $Delta u_2$')
    plt.plot(time_axis, dU_ml_arr[:, 0], 'c--', linewidth=2, label='Supervised Learning: $Delta u_1$')
    plt.plot(time_axis, dU_ml_arr[:, 1], 'm--', linewidth=2, label='Supervised Learning: $Delta u_2$')

    plt.title('Direct Output Parameter Comparison ($Delta u$)', fontsize=14)
    plt.xlabel('Time (sec)', fontsize=12)
    plt.ylabel('Control Increments ($Delta u$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()