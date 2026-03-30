import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import StateSpace
from qpsolvers import solve_qp
from prob3_learn import MPCNet 

def solve_kkt_from_active_set(H, q, G, h, active_mask):
    act_idx = np.where(active_mask == 1)[0]
    
    if len(act_idx) == 0:
        return np.linalg.solve(H, -q)
        
    G_act = G[act_idx]
    h_act = h[act_idx]
    
    n_u = H.shape[0]
    n_act = G_act.shape[0]
    
    KKT_left = np.block([
        [H, G_act.T],
        [G_act, np.zeros((n_act, n_act))]
    ])
    KKT_right = np.concatenate([-q, h_act])
    
    try:
        sol = np.linalg.solve(KKT_left, KKT_right)
        return sol[:n_u] 
    except np.linalg.LinAlgError:
        return np.zeros(n_u)


def main():
    model_filename = 'prob3_model.pth'
    nn_model = MPCNet()
    
    try:
        nn_model.load_state_dict(torch.load(model_filename))
        nn_model.eval()
        print(f"'{model_filename}' Classification Model loaded.")
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

    P_qp = H.astype(np.float64)
    G_qp = G.astype(np.float64)

    A_c = np.array([[-0.01, 0], [0, -0.01]])
    B_c = np.array([[0.4, -0.5], [-0.3, 0.4]])
    C_c = np.eye(2)
    D_c = np.zeros((2, 2))
    
    sys_c = StateSpace(A_c, B_c, C_c, D_c)
    sys_d = sys_c.to_discrete(2.0)
    Ad, Bd, Cd = sys_d.A, sys_d.B, sys_d.C

    sim_steps = 300  
    time_axis = np.arange(sim_steps) * 2.0 

    R_ref_history = np.zeros((sim_steps, 2))
    R_ref_history[50:, 0] = 0.63
    R_ref_history[50:, 1] = 0.79

    x_exact, u_prev_exact = np.zeros(2), np.zeros(2)
    x_ml, u_prev_ml = np.zeros(2), np.zeros(2)
    
    Y_exact_history, U_exact_history = [], []
    Y_ml_history, U_ml_history = [], []
    
    True_Active_history = []
    ML_Active_history = []

    print("Simulation starts...")

    for k in range(sim_steps):
        r_ref = R_ref_history[k]
        
        # ---  Exact MPC  ---
        p_exact = np.concatenate((x_exact, u_prev_exact, r_ref))
        q_qp = (F.T @ p_exact).astype(np.float64)
        h_qp = (W + E @ p_exact).astype(np.float64)
        
        try:
            dU_exact = solve_qp(P_qp, q_qp, G_qp, h_qp, solver='quadprog')
            if dU_exact is None: dU_exact = np.zeros(2)
        except:
            dU_exact = np.zeros(2)
            
        u_exact = np.clip(u_prev_exact + dU_exact, -1.0, 1.0) 
        
        true_residual = G_qp @ dU_exact - h_qp
        true_active = (np.abs(true_residual) < 1e-4).astype(int)
        
        # --- Supervised Learning ---
        p_ml = np.concatenate((x_ml, u_prev_ml, r_ref))
        q_qp_ml = (F.T @ p_ml).astype(np.float64)
        h_qp_ml = (W + E @ p_ml).astype(np.float64)
        
        p_tensor = torch.FloatTensor(p_ml).unsqueeze(0)
        with torch.no_grad():
            active_probs = nn_model(p_tensor).numpy()[0]
            ml_active = (active_probs > 0.5).astype(int)
            
        dU_ml = solve_kkt_from_active_set(P_qp, q_qp_ml, G_qp, h_qp_ml, ml_active)
        u_ml = np.clip(u_prev_ml + dU_ml, -1.0, 1.0)
        

        Y_exact_history.append(Cd @ x_exact)
        Y_ml_history.append(Cd @ x_ml)
        U_exact_history.append(u_exact)
        U_ml_history.append(u_ml)
        
        True_Active_history.append(np.argmax(true_active) if np.sum(true_active) > 0 else -1)
        ML_Active_history.append(np.argmax(ml_active) if np.sum(ml_active) > 0 else -1)
        
        x_exact = Ad @ x_exact + Bd @ u_exact
        u_prev_exact = u_exact
        
        x_ml = Ad @ x_ml + Bd @ u_ml
        u_prev_ml = u_ml

    Y_exact, Y_ml = np.array(Y_exact_history), np.array(Y_ml_history)
    U_exact, U_ml = np.array(U_exact_history), np.array(U_ml_history)

    print("Result plotting...")
    plt.figure(figsize=(14, 12)) 

    # [Plot 1] Output (y) 
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, Y_exact[:, 0], 'b-', linewidth=2.5, label='Exact MPC: $y_1$')
    plt.plot(time_axis, Y_exact[:, 1], 'r-', linewidth=2.5, label='Exact MPC: $y_2$')
    plt.plot(time_axis, Y_ml[:, 0], 'c--', linewidth=2, label='Classification ML: $y_1$')
    plt.plot(time_axis, Y_ml[:, 1], 'm--', linewidth=2, label='Classification ML: $y_2$')

    plt.plot(time_axis, R_ref_history[:, 0], 'b:', linewidth=2, alpha=0.7, label='Ref $r_1$')
    plt.plot(time_axis, R_ref_history[:, 1], 'r:', linewidth=2, alpha=0.7, label='Ref $r_2$')

    plt.title('Output Responses (Step Tracking): Solver vs Classification ML', fontsize=14)
    plt.ylabel('Outputs ($y_1, y_2$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # [Plot 2] Control Input (u) 
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, U_exact[:, 0], 'b-', linewidth=2.5, label='Exact MPC: $u_1$')
    plt.plot(time_axis, U_exact[:, 1], 'r-', linewidth=2.5, label='Exact MPC: $u_2$')
    plt.plot(time_axis, U_ml[:, 0], 'c--', linewidth=2, label='Classification ML: $u_1$')
    plt.plot(time_axis, U_ml[:, 1], 'm--', linewidth=2, label='Classification ML: $u_2$')

    plt.axhline(1.0, color='k', linestyle='-', alpha=0.8, label='Max Limit (+1)')
    plt.axhline(-1.0, color='k', linestyle='-', alpha=0.8, label='Min Limit (-1)')

    plt.title('Control Inputs', fontsize=14)
    plt.ylabel('Control Inputs ($u_1, u_2$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # [Plot 3] Active Set 
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, True_Active_history, 'c--', linewidth=2, label='Solver Active Set')
    plt.plot(time_axis, ML_Active_history, 'm--', linewidth=2, label='Predicted Active Set (Classification)')
    
    plt.yticks([-1, 0, 1, 2, 3], ['Unconstrained', '$u_1$ Max Limit', '$u_1$ Min Limit', '$u_2$ Max Limit', '$u_2$ Min Limit'])
    plt.title('Active Set Classification Accuracy', fontsize=14)
    plt.xlabel('Time (sec)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()