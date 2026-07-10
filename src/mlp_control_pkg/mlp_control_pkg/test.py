import omni.usd
import omni.timeline
from pxr import UsdGeom, Gf, Usd
import numpy as np
import math

class State:
    def __init__(self):
        self.curve_points_world = None
        self.car_path = None
        self.curve_path = None
        self.initialized = False
        self.frame_count = 0
        self.last_known_index = 0
        self.path_direction = 1

if not hasattr(omni.graph, '_curve_tracker_state'):
    omni.graph._curve_tracker_state = State()

state = omni.graph._curve_tracker_state

def setup(db):
    state = db.per_instance_state
    state.initialized = False
    state.last_known_index = 0
    state.path_direction = 1

def compute(db):
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    current_time = Usd.TimeCode(timeline.get_current_time())
    
    # === INITIALIZATION ===
    if not state.initialized:
        selection = omni.usd.get_context().get_selection().get_selected_prim_paths()
        
        if len(selection) < 2 and (state.car_path is None or state.curve_path is None):
            if state.frame_count % 120 == 0: 
                print("WAITING: Select Car and Curve...")
            state.frame_count += 1
            return False
        
        if len(selection) >= 2:
            prim_a = stage.GetPrimAtPath(selection[0])
            prim_b = stage.GetPrimAtPath(selection[1])
            if prim_a.IsA(UsdGeom.BasisCurves):
                state.curve_path = selection[0]
                state.car_path = selection[1]
            elif prim_b.IsA(UsdGeom.BasisCurves):
                state.curve_path = selection[1]
                state.car_path = selection[0]
        
        if state.curve_path and state.car_path:
            curve_prim = stage.GetPrimAtPath(state.curve_path)
            curve_geom = UsdGeom.BasisCurves(curve_prim)
            points = curve_geom.GetPointsAttr().Get()
            xform = UsdGeom.Xformable(curve_prim)
            world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            
            curve_points_world = []
            for pt in points:
                wp = world_transform.Transform(Gf.Vec3d(pt[0], pt[1], pt[2]))
                curve_points_world.append(np.array(wp))
            
            state.curve_points_world = np.array(curve_points_world)
            
            # === DETERMINE INITIAL PATH DIRECTION ===
            car_prim = stage.GetPrimAtPath(state.car_path)
            car_xform = UsdGeom.Xformable(car_prim)
            car_mat = car_xform.ComputeLocalToWorldTransform(current_time)
            car_pos = car_mat.ExtractTranslation()
            car_pos_np = np.array([car_pos[0], car_pos[1], car_pos[2]])
            car_rot = car_mat.ExtractRotationMatrix()
            car_fwd = np.array([car_rot[0][0], car_rot[0][1], car_rot[0][2]])
            car_angle_rad = math.atan2(car_fwd[1], car_fwd[0])
            
            dists = np.linalg.norm(state.curve_points_world - car_pos_np, axis=1)
            min_idx = np.argmin(dists)
            
            next_idx = (min_idx + 1) % len(state.curve_points_world)
            prev_idx = (min_idx - 1) % len(state.curve_points_world)
            
            forward_tangent = state.curve_points_world[next_idx] - state.curve_points_world[min_idx]
            backward_tangent = state.curve_points_world[min_idx] - state.curve_points_world[prev_idx]
            
            forward_angle = math.atan2(forward_tangent[1], forward_tangent[0])
            backward_angle = math.atan2(backward_tangent[1], backward_tangent[0])
            
            forward_diff = abs(forward_angle - car_angle_rad)
            if forward_diff > math.pi: forward_diff = 2 * math.pi - forward_diff
                
            backward_diff = abs(backward_angle - car_angle_rad)
            if backward_diff > math.pi: backward_diff = 2 * math.pi - backward_diff
            
            if forward_diff < backward_diff:
                state.path_direction = 1
                print("Path direction: FORWARD")
            else:
                state.path_direction = -1
                print("Path direction: BACKWARD")
            
            state.initialized = True
            state.last_known_index = min_idx
            print(f"SUCCESS: Tracking {state.car_path}")

    if not state.initialized: return False

    # === GET CAR POS AND ORIENTATION ===
    car_prim = stage.GetPrimAtPath(state.car_path)
    if not car_prim: return False
    
    car_xform = UsdGeom.Xformable(car_prim)
    car_mat = car_xform.ComputeLocalToWorldTransform(current_time)
    car_pos = car_mat.ExtractTranslation()
    car_pos_np = np.array([car_pos[0], car_pos[1], car_pos[2]])
    
    car_rot = car_mat.ExtractRotationMatrix()
    car_fwd = np.array([car_rot[0][0], car_rot[0][1], car_rot[0][2]])
    car_angle_rad = math.atan2(car_fwd[1], car_fwd[0])
    car_angle_deg = math.degrees(car_angle_rad)
    
    # === 1. CYCLIC SEARCH OPTIMIZATION ===
    num_points = len(state.curve_points_world)
    SEARCH_RADIUS = 20 
    
    search_indices = np.arange(state.last_known_index - SEARCH_RADIUS, state.last_known_index + SEARCH_RADIUS + 1)
    search_indices = search_indices % num_points
    
    search_subset = state.curve_points_world[search_indices]
    
    if len(search_subset) > 0:
        dists = np.linalg.norm(search_subset - car_pos_np, axis=1)
        local_min_idx = np.argmin(dists)
        min_idx = search_indices[local_min_idx] 
    else:
        min_idx = 0

    state.last_known_index = min_idx

    # === 1.5 FIND EXACT CONTINUOUS CLOSEST POINT ===
    p_curr = state.curve_points_world[min_idx]
    next_idx = (min_idx + state.path_direction) % num_points
    p_next = state.curve_points_world[next_idx]
    
    # Vector of the line segment
    segment_vec = p_next - p_curr
    segment_len_sq = np.dot(segment_vec, segment_vec)
    
    if segment_len_sq > 0:
        # Vector from current vertex to car
        v_to_car = car_pos_np - p_curr
        
        # Project car vector onto segment vector to find normalized distance 't'
        t = np.dot(v_to_car, segment_vec) / segment_len_sq
        
        # Clamp t between 0 and 1 so the point stays on the segment
        t = max(0.0, min(1.0, t))
        
        exact_closest_point = p_curr + t * segment_vec
    else:
        exact_closest_point = p_curr
    
    # === 2. CYCLIC TANGENT ===
    next_idx_tangent = (min_idx + state.path_direction) % num_points
    curve_tangent = state.curve_points_world[next_idx_tangent] - state.curve_points_world[min_idx]
    if state.path_direction == -1:
        curve_tangent = -curve_tangent

    # === SMOOTH OFFSET ===
    v_to_car = car_pos_np - state.curve_points_world[min_idx]
    tangent_norm = np.linalg.norm(curve_tangent)
    if tangent_norm > 0:
        unit_tangent = curve_tangent / tangent_norm
        projection_dist = np.dot(v_to_car, unit_tangent)
    else:
        projection_dist = 0.0

    # === 3. MULTI-POINT LOOKAHEAD ===
    # Lookahead distances: 0.1 + projection, 10 points along the curve
    base_dist = 0.1
    target_distances = []
    for i in range(10):
        target_distances.append(base_dist* (i + 1) + projection_dist)  # Increment by 0.1 for each subsequent point

    # Structure to hold (position_vector, tangent_vector)
    found_targets = []
    
    current_idx = min_idx
    accumulated_dist = 0.0
    targets_found_count = 0
    
    while targets_found_count < 10:
        next_idx = (current_idx + state.path_direction) % num_points
        
        p_curr = state.curve_points_world[current_idx]
        p_next = state.curve_points_world[next_idx]
        
        segment_dist = np.linalg.norm(p_next - p_curr)
        
        # Determine tangent for this segment
        segment_tangent = p_next - p_curr
        if state.path_direction == -1:
            segment_tangent = -segment_tangent
            
        while targets_found_count < 10:
            target_d = target_distances[targets_found_count]
            
            if accumulated_dist + segment_dist >= target_d:
                remaining_dist = target_d - accumulated_dist
                ratio = remaining_dist / segment_dist if segment_dist > 0 else 0
                
                # Interpolate position
                pos = p_curr + (p_next - p_curr) * ratio
                
                # Store position and the specific tangent at this point
                found_targets.append({
                    "pos": pos,
                    "tangent": segment_tangent
                })
                
                targets_found_count += 1
            else:
                break 
        
        accumulated_dist += segment_dist
        current_idx = next_idx

        # Safety break
        if accumulated_dist > target_distances[-1] + 100:
            break
            
    # Fallback if loop ends early
    while len(found_targets) < 10:
        found_targets.append({
            "pos": car_pos_np,
            "tangent": curve_tangent
        })

    # === UPDATE MARKERS ===
    colors = [(1, 1, 1)] * 10  # Default to white for all    
    
    for i in range(10):
        marker_path = f"/World/TargetPoint_{i}"
        marker_prim = stage.GetPrimAtPath(marker_path)
        
        if not marker_prim.IsValid():
            marker_geom = UsdGeom.Sphere.Define(stage, marker_path)
            marker_geom.GetRadiusAttr().Set(0.1) 
            marker_geom.GetDisplayColorAttr().Set([colors[i]]) 
            marker_prim = marker_geom.GetPrim()

        xform = UsdGeom.Xformable(marker_prim)
        xform.ClearXformOpOrder()
        pos = found_targets[i]["pos"]
        xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), 0.0))

    # === OUTPUT CONSTRUCTION ===
    # Start with the index
    output_list = [float(min_idx)]
    
    # Calculate pos and absolute heading for each target
    for i in range(10):
        target = found_targets[i]
        pos = target["pos"]
        tangent = target["tangent"]
        
        # 1. 計算該點的絕對角度 (弧度轉角度)
        path_angle_rad = math.atan2(tangent[1], tangent[0])
        path_angle_deg = math.degrees(path_angle_rad)
        
        # 2. 將角度強制正規化到 0 ~ 360 度
        # 如果你希望範圍是 -180 ~ 180，則使用下面的註解代碼
        abs_heading = path_angle_deg % 360.0 
        
        output_list.append(float(pos[0]))    # x
        output_list.append(float(pos[1]))    # y
        output_list.append(float(abs_heading)) # 絕對角度 (不再是差值)

    # Final Output: [idx, x1, y1, angle1, x2, y2, angle2, ..., x10, y10, angle10]
    db.outputs.output_data = output_list
    
    return True

def cleanup(db):
    state.initialized = False