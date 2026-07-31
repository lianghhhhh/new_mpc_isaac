import json
import torch
import joblib
import numpy as np
import casadi as ca
import l4casadi as l4c
from mlp_control_pkg.car_dynamic_model import CarDynamicModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
torch.backends.mkldnn.enabled = False

def loadConfig(config_path='config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# ---------------------------------------------------------------------------
# Body-frame delta helpers -- must exactly match convention 'B' from the
# training-side utils.py (state_delta_body_frame / apply_body_frame_delta
# with BODY_FRAME_CONVENTION = 'B'), since that's what the currently
# deployed model was trained on. If you retrain with a different
# convention, update these two functions to match, or the model's
# position predictions will be silently rotated into the wrong frame.
#
# STATE_COLS layout (7-dim physical state): [pos_x, pos_y, sin_theta,
# cos_theta, vx, vy, wz]. Index 0/1 hold body-frame (forward, lateral)
# deltas instead of raw world dx/dy for anything derived by these
# functions -- everything else (sin/cos, vx, vy, wz) is a plain delta.
# ---------------------------------------------------------------------------

def compute_body_delta(state_from, state_to):
    """CasADi-symbolic equivalent of the training-side state_delta_body_frame
    (convention B). state_from/state_to are 7x1 MX vectors."""
    delta = state_to - state_from
    sin_h = state_from[2]
    cos_h = state_from[3]
    dx = delta[0]
    dy = delta[1]
    forward = sin_h * dx + cos_h * dy
    lateral = -cos_h * dx + sin_h * dy
    return ca.vertcat(forward, lateral, delta[2], delta[3], delta[4], delta[5], delta[6])


def apply_body_delta(state_from, delta):
    """CasADi-symbolic equivalent of the training-side apply_body_frame_delta
    (convention B). Does NOT renormalize sin/cos -- caller must do that
    afterwards, same as in training."""
    sin_h = state_from[2]
    cos_h = state_from[3]
    forward = delta[0]
    lateral = delta[1]
    dx = sin_h * forward - cos_h * lateral
    dy = cos_h * forward + sin_h * lateral
    return ca.vertcat(
        state_from[0] + dx, state_from[1] + dy,
        state_from[2] + delta[2], state_from[3] + delta[3],
        state_from[4] + delta[4], state_from[5] + delta[5], state_from[6] + delta[6],
    )

class FastDynamicsModel(torch.nn.Module):
    """
    Expects exactly the 140-element window (10 steps x 14 features).
    Only returns the final predicted delta (7 features) to keep the graph small.
    """
    def __init__(self, base_model, scaler_in, scaler_out):
        super().__init__()
        self.base = base_model
        self.register_buffer('in_mean', torch.tensor(scaler_in.mean_, dtype=torch.float32))
        self.register_buffer('in_scale', torch.tensor(scaler_in.scale_, dtype=torch.float32))
        self.register_buffer('out_mean', torch.tensor(scaler_out.mean_, dtype=torch.float32))
        self.register_buffer('out_scale', torch.tensor(scaler_out.scale_, dtype=torch.float32))

    def forward(self, nn_input):
        B = nn_input.shape[0]
        nn_input = nn_input.view(B, 10, 14)

        # Scale (per-feature stats broadcast across the 10 timesteps)
        x_scaled = (nn_input - self.in_mean) / self.in_scale

        # Flatten for the MLP
        x_scaled = x_scaled.reshape(B, -1)

        # Run MLP
        y_scaled = self.base(x_scaled)

        # Unscale output
        delta_pred = (y_scaled * self.out_scale) + self.out_mean
        return delta_pred

def loadModelFunc(model_path, dt):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = 'cpu'

    base_model = CarDynamicModel(input_size=140, output_size=7)
    base_model.load_state_dict(torch.load(f'{model_path}/best.pth', map_location=device))
    
    scaler_in = joblib.load(f'{model_path}/scaler_in.pkl')
    scaler_out = joblib.load(f'{model_path}/scaler_out.pkl')

    model = FastDynamicsModel(base_model, scaler_in, scaler_out)
    model.to(device)
    model.eval()
    
    l4c_model = l4c.L4CasADi(model, device=device, name='car_model')
    
    # Input is exactly the 140 features required for the LSTM
    nn_input_sym = ca.MX.sym('nn_input', 140) 
    
    # Transpose for PyTorch mapping
    delta_sym = l4c_model(nn_input_sym.T).T
    
    nn_model_func = ca.Function('nn_model_func', [nn_input_sym], [delta_sym])
    return nn_model_func, l4c_model.shared_lib_dir, l4c_model.name

def createAcadosSolver(nn_model_func, lib_dir, lib_name, N, dt):
    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = 'car_model'

    # ---------------------------------------------------------
    # 1. Augmented State and Parameters
    # ---------------------------------------------------------
    # State includes 10 past states (10 * 7 = 70) and 9 past controls (9 * 2 = 18)
    nx = 88
    nu = 2
    x = ca.MX.sym('x', nx)  
    u = ca.MX.sym('u', nu)  
    
    # Parameters NOW ONLY hold the 4 target path variables (x, y, sin, cos)
    np_p = 5
    p = ca.MX.sym('p', np_p)
    target_path = p[0:4]
    v_ref = p[4]

    # Extract rolling history from the augmented state vector
    X_elements = [x[i*7 : (i+1)*7] for i in range(10)]
    U_elements = [x[70 + i*2 : 70 + (i+1)*2] for i in range(9)]

    # ---------------------------------------------------------
    # 2. Build the 140 feature vector INSIDE CasADi
    # ---------------------------------------------------------
    # Append the CURRENT optimization control `u` to complete the 10-step control sequence
    U_full = U_elements + [u]
    X_full = X_elements

    nn_inputs = []
    for i in range(10):
        ctrl = U_full[i]
        state_i = X_full[i]
        
        # 5 dim abs state (strip x,y -- matches ABS_STATE_MASK in training utils.py)
        abs_state = state_i[2:] 
        
        # 7 dim delta state, body-frame (forward/lateral) position channels
        delta_state = ca.MX.zeros(7, 1) if i == 0 else compute_body_delta(X_full[i - 1], X_full[i])
        
        nn_inputs.append(ca.vertcat(ctrl, abs_state, delta_state))

    flat_nn_input = ca.vertcat(*nn_inputs) # 140x1
    
    # Run the neural network
    delta_pred = nn_model_func(flat_nn_input)

    # ---------------------------------------------------------
    # 3. Next State Integration & Shifting
    # ---------------------------------------------------------
    current_physical_state = X_full[-1]
    # delta_pred's position channels are (forward, lateral) in the body
    # frame of current_physical_state -- rotate back to world frame before
    # integrating, same as apply_body_frame_delta does in training/rollout.
    next_physical_state = apply_body_delta(current_physical_state, delta_pred)
    
    # Renormalize geometric limits securely
    ns_list = [next_physical_state[j] for j in range(7)]
    pairs = [(2,3)]
    for s_idx, c_idx in pairs:
        norm = ca.sqrt(next_physical_state[s_idx]**2 + next_physical_state[c_idx]**2 + 1e-8)
        ns_list[s_idx] = next_physical_state[s_idx] / norm
        ns_list[c_idx] = next_physical_state[c_idx] / norm
    
    x_next_new = ca.vertcat(*ns_list)

    # SHIFT THE WINDOW: Drop the oldest state/control, append the new ones
    X_next_aug = ca.vertcat(x[7:70], x_next_new)
    U_next_aug = ca.vertcat(x[72:88], u)
    
    # Final augmented next state
    x_next = ca.vertcat(X_next_aug, U_next_aug)

    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = x_next
    ocp.model = model

    # ---------------------------------------------------------
    # 4. Cost Function (NONLINEAR_LS for Speed)
    # ---------------------------------------------------------
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    current_x_phys = x[63:70] # The current physical state

    prev_u_hist = x[86:88]
    delta_u = u - prev_u_hist

    vel_x, vel_y = current_x_phys[4], current_x_phys[5]
    speed = ca.sqrt(vel_x**2 + vel_y**2 + 1e-6)
    speed_error = speed - v_ref

    tx, ty = target_path[2], target_path[3]     # sin(path heading), cos(path heading)
    dx = current_x_phys[0] - target_path[0]
    dy = current_x_phys[1] - target_path[1]

    # CORRECTED for Convention B (Y-forward):
    # forward = sin*dx + cos*dy
    # lateral = -cos*dx + sin*dy
    along_track_error = dx * tx + dy * ty        # along path direction (loose)
    cross_track_error = -dx * ty + dy * tx       # perpendicular to path (tight)

    state_error = ca.vertcat(
        cross_track_error,
        along_track_error,
        current_x_phys[2] - target_path[2],
        current_x_phys[3] - target_path[3],
    )

    ocp.cost.cost_type_0 = 'NONLINEAR_LS'
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # y_expr is the vector of residuals:
    # [error_x, error_y, error_sin, error_cos, speed_error, u_fl, u_fr, u_rl, u_rr, du_fl, du_fr, du_rl, du_rr]
    ocp.model.cost_y_expr_0 = ca.vertcat(state_error, speed_error, u, delta_u)
    ocp.model.cost_y_expr = ca.vertcat(state_error, speed_error, u, delta_u)
    ocp.model.cost_y_expr_e = state_error

    # 定義各個變數的「最大容許誤差 / 最大預期範圍」
    max_cte   = 0.15  # 容許最大橫向誤差 (公尺)
    max_ate   = 0.5   # 容許最大縱向誤差 (公尺)
    max_head  = 0.2   # 容許最大航向誤差 (sin/cos 差值)
    max_speed = 1.0   # 容許最大速度誤差 (m/s)
    max_u     = 5.0   # 控制輸出的物理極限
    max_du    = 1.0   # 容許的單步最大控制變化量

    norm_factors = np.array([
        max_cte**2, max_ate**2, max_head**2, max_head**2,
        max_speed**2, max_u**2, max_u**2, max_du**2, max_du**2
    ])

    w_cte   = 10.0
    w_ate   = 3.0
    w_head  = 8.0
    w_speed = 10.0
    w_u     = 2.5
    w_du    = 0.1

    intuitive_weights = np.array([
        w_cte, w_ate, w_head, w_head, w_speed, w_u, w_u, w_du, w_du
    ])

    # 最終傳給 Acados 的權重矩陣 = 直覺權重 / 正規化分母
    final_weights = intuitive_weights / norm_factors
    W_stage = np.diag(final_weights)
    
    # Terminal cost
    terminal_intuitive_weights = np.array([w_cte * 1.5, w_ate * 1.5, w_head * 1.5, w_head * 1.5])
    final_terminal_weights = terminal_intuitive_weights / norm_factors[:4]
    W_terminal = np.diag(final_terminal_weights)

    ocp.cost.W_0 = W_stage
    ocp.cost.W = W_stage
    ocp.cost.W_e = W_terminal

    # References (We already subtracted the target inside cost_y_expr, so the target residual is exactly zero)
    ocp.cost.yref_0 = np.zeros(9)
    ocp.cost.yref = np.zeros(9)
    ocp.cost.yref_e = np.zeros(4)

    # ---------------------------------------------------------
    # 5. Constraints & Options
    # ---------------------------------------------------------
    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(np_p)
    
    ocp.constraints.lbu = np.array([-5.0, -5.0])
    ocp.constraints.ubu = np.array([5.0, 5.0])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.model_external_shared_lib_dir = lib_dir
    ocp.solver_options.model_external_shared_lib_name = lib_name
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.qp_tol = 1e-3

    # Force GCC/Clang to compile the generated solver with high optimization
    import os
    os.environ['CFLAGS'] = '-O3 -ffast-math -march=native'
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return acados_solver