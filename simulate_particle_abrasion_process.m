function geological_abrasion_ultimate_dashboard()
    % =========================================================================
    % 颗粒地质磨蚀全动态仿真工作站 (终极完整版)
    % 特性：20颗粒全阵列 + 统计包络带 + 时间轴回溯 + 低面数绝对闭合流形引擎
    % =========================================================================
    
    % --- 1. 核心物理与批量演化参数(ueser-defined) ---
    num_particles = 20;          % 阵列颗粒总数
    scale_variance = 0.20;       % 长宽高缩放扰动系数 (±10%)
    base_scale_xyz = [1.0, 1.2, 0.6]; 
    
    num_faces = 5;               % 初始面数 (利用半空间截断法，设为4也能绝对闭合)
    total_cycles = 500;          % 总磨蚀步数
    abrasion_rate = 0.05;        % 宏观去棱角速度
    
    D9_15_start = 0.030;         % 初始微观粗糙度
    roughness_decay_rate = 0.005;% 粗糙度抛光衰减率 (\lambda)
    
    mesh_level = 4;              % 网格精细度
    random_seed = 2025;          
    
    rng(random_seed);
    
    % --- 2. 预分配海量数据存储矩阵 ---
    history_irr_all  = zeros(num_particles, total_cycles); 
    history_sph_all  = zeros(num_particles, total_cycles);
    history_mps_all  = zeros(num_particles, total_cycles); 
    history_vol_all  = zeros(num_particles, total_cycles);
    history_area_all = zeros(num_particles, total_cycles);
    history_EI_all   = zeros(num_particles, total_cycles);
    history_FI_all   = zeros(num_particles, total_cycles);
    history_d915_all = zeros(num_particles, total_cycles);
    history_conv_all = zeros(num_particles, total_cycles); 
    
    % 存储 20 个颗粒在 500 帧中的所有 3D 顶点数据
    V_render_history_all = cell(num_particles, total_cycles); 
    
    % --- 3. 预计算拓扑算子 (提速核心) ---
    fprintf('1. 正在初始化基础测地网格与拓扑算子...\n');
    [V_sphere, F] = create_subdivided_sphere(mesh_level);
    E1 = [F(:,1); F(:,2); F(:,3)]; E2 = [F(:,2); F(:,3); F(:,1)];
    A_mat = sparse(E1, E2, 1, size(V_sphere,1), size(V_sphere,1));
    A_mat = A_mat | A_mat'; 
    invD = spdiags(1./sum(A_mat,2), 0, size(V_sphere,1), size(V_sphere,1));
    L_op = invD * A_mat; 
    
    % --- 4. 核心批量预计算循环 ---
    fprintf('2. 正在后台演算 %d 个颗粒的完整物理演化史...\n', num_particles);
    
    for p = 1:num_particles
        current_scale = base_scale_xyz .* (1 + scale_variance * (rand(1, 3) - 0.5));
        
        % ★ 半空间相交法：保证极低面数下也能生成完美闭合多面体
        is_closed_polyhedron = false;
        while ~is_closed_polyhedron
            pts = randn(num_faces, 3);
            pts = pts ./ sqrt(sum(pts.^2, 2));
            
            % 静电斥力分布算法
            for iter = 1:15
                force = zeros(num_faces, 3);
                for i = 1:num_faces
                    for j = 1:num_faces
                        if i ~= j
                            v = pts(i,:) - pts(j,:);
                            d2 = sum(v.^2);
                            force(i,:) = force(i,:) + v / (d2 + 0.01);
                        end
                    end
                end
                pts = pts + 0.1 * force;
                pts = pts ./ sqrt(sum(pts.^2, 2));
            end
            
            face_dist = 1.0 + 0.6 * rand(num_faces, 1);
            r_base = inf(size(V_sphere, 1), 1);
            for i = 1:num_faces
                dot_RN = V_sphere * pts(i, :)';
                t = face_dist(i) ./ dot_RN;
                t(dot_RN <= 0) = inf;
                r_base = min(r_base, t);
            end
            
            % 验证完全闭合
            if max(r_base) < inf
                is_closed_polyhedron = true;
            end
        end
        
        V_base = V_sphere .* r_base; 
        V_base = V_base .* current_scale;
        V_base = V_base - mean(V_base, 1); 
        
        initial_volume = compute_volume(V_base, F);
        initial_area   = compute_area(V_base, F);
        
        % 生成 3D 微观高频纹理场
        dx_sh = zeros(size(V_sphere, 1), 1); dy_sh = zeros(size(V_sphere, 1), 1); dz_sh = zeros(size(V_sphere, 1), 1);
        num_waves = 80; base_frequency = 15; 
        for w = 1:num_waves
            k_dir = randn(1, 3); k_dir = k_dir / norm(k_dir);
            freq = base_frequency + rand() * 8; phase = rand() * 2 * pi;
            wave_val = sin(freq * (V_base * k_dir') + phase); 
            amp_dir = randn(1, 3); amp_dir = amp_dir / norm(amp_dir);
            dx_sh = dx_sh + amp_dir(1) * wave_val; dy_sh = dy_sh + amp_dir(2) * wave_val; dz_sh = dz_sh + amp_dir(3) * wave_val;
        end
        V_roughness_unit = [dx_sh, dy_sh, dz_sh] / mean(sqrt(dx_sh.^2 + dy_sh.^2 + dz_sh.^2));
        
        % 正向演化推演
        V_current = V_base;
        for t = 1:total_cycles
            V_avg = L_op * V_current;
            V_current = (1 - abrasion_rate) * V_current + abrasion_rate * V_avg;
            curr_D9_15 = D9_15_start * exp(-roughness_decay_rate * t);
            
            V_render = V_current + V_roughness_unit * curr_D9_15;
            V_render_history_all{p, t} = V_render;
            
            C_mass = mean(V_render, 1);
            r_dist = sqrt(sum((V_render - C_mass).^2, 2));
            history_irr_all(p, t) = std(r_dist) / mean(r_dist);              
            
            curr_vol = compute_volume(V_current, F);
            curr_area = compute_area(V_current, F);
            history_sph_all(p, t) = (pi^(1/3) * (6 * curr_vol)^(2/3)) / curr_area; 
            
            V_centered = V_current - C_mass;
            cov_mat = (V_centered' * V_centered) / size(V_centered, 1); 
            eigenvals = sort(eig(cov_mat), 'descend');                  
            L_axis = sqrt(eigenvals(1)); I_axis = sqrt(eigenvals(2)); S_axis = sqrt(eigenvals(3));                                
            history_mps_all(p, t) = (S_axis^2 / (L_axis * I_axis))^(1/3); 
            
            [~, hull_vol] = convhull(V_current(:,1), V_current(:,2), V_current(:,3));
            history_conv_all(p, t) = curr_vol / hull_vol;
            
            history_vol_all(p, t) = curr_vol / initial_volume; 
            history_area_all(p, t) = curr_area / initial_area;
            history_EI_all(p, t) = I_axis / L_axis; 
            history_FI_all(p, t) = S_axis / I_axis;
            history_d915_all(p, t) = curr_D9_15; 
        end
        fprintf('  - 演化组 %d / %d 计算完毕...\n', p, num_particles);
    end
    
    % --- 5. 建立可视化仪表盘 UI ---
    fprintf('3. 启动全景阵列仿真监控台...\n');
    hFig = figure('Color', 'w', 'Name', 'Time-Scrubbing Particle Evolution Dashboard', 'Position', [50, 100, 1600, 750]);
    
    % ★ 全局时间轴滑块
    slider_ui = uicontrol('Style', 'slider', 'Min', 1, 'Max', total_cycles, 'Value', 1, ...
        'Units', 'normalized', 'Position', [0.15 0.02 0.7 0.03], ...
        'SliderStep', [1/(total_cycles-1), 10/(total_cycles-1)]);
        
    txt_slider = uicontrol('Style', 'text', 'String', 'Cycle: 1 / 500', ...
        'Units', 'normalized', 'Position', [0.45 0.05 0.1 0.03], ...
        'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'w');
                      
    % --- 3D 颗粒阵列排布设计 (5x4) ---
    ax_3d = subplot(2, 4, [1, 5]);
    hold(ax_3d, 'on');
    grid_spacing = 5.5; 
    grid_cols = 5;      
    offsets = zeros(num_particles, 3);
    h_surf_all = gobjects(1, num_particles);
    
    for p = 1:num_particles
        row = floor((p-1)/grid_cols);
        col = mod(p-1, grid_cols);
        offsets(p, :) = [(col - (grid_cols-1)/2)*grid_spacing, -(row - 1.5)*grid_spacing, 0];
        V_init = V_render_history_all{p, 1} + offsets(p, :);
        h_surf_all(p) = trisurf(F, V_init(:,1), V_init(:,2), V_init(:,3), ...
            'Parent', ax_3d, 'FaceColor', [0.4 0.6 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.95);
    end
    hold(ax_3d, 'off');
    
    material(ax_3d, 'dull'); camlight(ax_3d, 'headlight'); camlight(ax_3d, 'left'); 
    lighting(ax_3d, 'gouraud'); axis(ax_3d, 'equal', 'off'); view(ax_3d, [30, 45]);
    xlim(ax_3d, [-9 9]); ylim(ax_3d, [-7 7]); zlim(ax_3d, [-3 3]);
    title_3d = title(ax_3d, sprintf('Cycles: 1 | %d Particles Array', num_particles), 'FontName', 'Times New Roman', 'FontSize', 15, 'FontWeight', 'bold');
    
    % --- 雷达图表初始化 ---
    ax_irr = subplot(2, 4, 2); hold(ax_irr, 'on');
    h_patch_irr = fill(ax_irr, NaN, NaN, 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    h_line_irr = plot(ax_irr, NaN, NaN, 'r-', 'LineWidth', 2, 'DisplayName', 'Mean');
    setup_2d_axes(ax_irr, '1. Irregularity (\sigma_r / \mu_r)', [0 total_cycles], [0, 0.6]);
    
    ax_sph = subplot(2, 4, 6); hold(ax_sph, 'on');
    h_patch_sph = fill(ax_sph, NaN, NaN, 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    h_line_sph = plot(ax_sph, NaN, NaN, 'g-', 'LineWidth', 2, 'DisplayName', 'True (\Psi)');
    color_mps = [0.466 0.674 0.188];
    h_patch_mps = fill(ax_sph, NaN, NaN, color_mps, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    h_line_mps = plot(ax_sph, NaN, NaN, '-', 'Color', color_mps, 'LineWidth', 2, 'DisplayName', 'MPS (\Psi_p)');
    setup_2d_axes(ax_sph, '2. Sphericity (\Psi & \Psi_p)', [0 total_cycles], [0.5, 1.0]);
    legend(ax_sph, 'Location', 'southeast', 'FontName', 'Times New Roman'); 
    
    ax_ret = subplot(2, 4, 3); hold(ax_ret, 'on');
    h_patch_vol = fill(ax_ret, NaN, NaN, 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    h_line_vol = plot(ax_ret, NaN, NaN, 'b-', 'LineWidth', 2, 'DisplayName', 'Volume');
    color_area = [0.85 0.33 0.10];
    h_patch_area = fill(ax_ret, NaN, NaN, color_area, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    h_line_area = plot(ax_ret, NaN, NaN, '-', 'Color', color_area, 'LineWidth', 2, 'DisplayName', 'Area');
    setup_2d_axes(ax_ret, '3. Retention (V/V_0, A/A_0)', [0 total_cycles], [0.3, 1.05]);
    legend(ax_ret, 'Location', 'northeast', 'FontName', 'Times New Roman'); 
    
    ax_zingg = subplot(2, 4, 7); hold(ax_zingg, 'on');
    h_patch_EI = fill(ax_zingg, NaN, NaN, 'm', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    h_line_EI = plot(ax_zingg, NaN, NaN, 'm-', 'LineWidth', 2, 'DisplayName', 'EI (I/L)');
    h_patch_FI = fill(ax_zingg, NaN, NaN, 'c', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    h_line_FI = plot(ax_zingg, NaN, NaN, 'c-', 'LineWidth', 2, 'DisplayName', 'FI (S/I)');
    setup_2d_axes(ax_zingg, '4. Zingg Indices', [0 total_cycles], [0.3, 1.0]);
    legend(ax_zingg, 'Location', 'best', 'FontName', 'Times New Roman'); 
    
    ax_d915 = subplot(2, 4, 4); hold(ax_d915, 'on');
    h_patch_d915 = fill(ax_d915, NaN, NaN, 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    h_line_d915 = plot(ax_d915, NaN, NaN, 'k-', 'LineWidth', 2);
    setup_2d_axes(ax_d915, '5. Micro-Roughness Decay (D_{9-15})', [0 total_cycles], [0, D9_15_start*1.1]);
    
    ax_conv = subplot(2, 4, 8); hold(ax_conv, 'on');
    color_conv = [0.494 0.184 0.556];
    h_patch_conv = fill(ax_conv, NaN, NaN, color_conv, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    h_line_conv = plot(ax_conv, NaN, NaN, '-', 'Color', color_conv, 'LineWidth', 2);
    setup_2d_axes(ax_conv, '6. Convexity (V / V_{hull})', [0 total_cycles], [0.85, 1.02]);
    
    pause(1.5); 
    
    % --- 6. 首次自动播放动画循环 ---
    for frame_t = 1:total_cycles
        if ~ishghandle(hFig), break; end
        set(slider_ui, 'Value', frame_t);
        update_visuals(frame_t);
    end
    
    % --- 7. 绑定拖拽回溯事件 ---
    if ishghandle(hFig)
        set(slider_ui, 'Callback', @(src, event) update_visuals(round(get(src, 'Value'))));
        fprintf('======================================\n');
        fprintf('全景阵列演化完成！\n请拖动界面底部的【进度滑块】，即可任意回放和锁定历史磨耗状态。\n');
    end

    % ========================================================
    % ★ 嵌套回调函数：时间回溯核心 UI 刷新逻辑 ★
    % ========================================================
    function update_visuals(t)
        % 1. 刷新 3D 阵列
        for p_idx = 1:num_particles
            set(h_surf_all(p_idx), 'Vertices', V_render_history_all{p_idx, t} + offsets(p_idx, :));
        end
        
        % 2. 刷新雷达图表截止时间带
        t_vec = 1:t;
        update_band(h_line_irr, h_patch_irr, t_vec, history_irr_all(:, 1:t));
        update_band(h_line_sph, h_patch_sph, t_vec, history_sph_all(:, 1:t));
        update_band(h_line_mps, h_patch_mps, t_vec, history_mps_all(:, 1:t));
        update_band(h_line_vol, h_patch_vol, t_vec, history_vol_all(:, 1:t));
        update_band(h_line_area, h_patch_area, t_vec, history_area_all(:, 1:t));
        update_band(h_line_EI, h_patch_EI, t_vec, history_EI_all(:, 1:t));
        update_band(h_line_FI, h_patch_FI, t_vec, history_FI_all(:, 1:t));
        update_band(h_line_d915, h_patch_d915, t_vec, history_d915_all(:, 1:t));
        update_band(h_line_conv, h_patch_conv, t_vec, history_conv_all(:, 1:t));
        
        % 3. 动态调整不规则度坐标轴自适应
        if t == 1
            ylim(ax_irr, [0, max(mean(history_irr_all(:,1))*1.15, 0.1)]); 
        end
        
        % 4. 更新文本显示
        set(title_3d, 'String', sprintf('Cycles: %d | %d Particles Array', t, num_particles), 'Color', 'k');
        set(txt_slider, 'String', sprintf('Cycle: %d / %d', t, total_cycles));
        
        drawnow;
    end
end

% ========================================================
% 子函数群：包络带计算、网格生成、体积与表面积计算
% ========================================================
function update_band(h_line, h_patch, t_vec, data_matrix)
    mean_val = mean(data_matrix, 1);
    std_val = std(data_matrix, 0, 1);
    upper_bound = mean_val + std_val;
    lower_bound = mean_val - std_val;
    
    set(h_line, 'XData', t_vec, 'YData', mean_val);
    if length(t_vec) > 1
        X_poly = [t_vec, fliplr(t_vec)];
        Y_poly = [upper_bound, fliplr(lower_bound)];
        set(h_patch, 'XData', X_poly, 'YData', Y_poly);
    else
        set(h_patch, 'XData', NaN, 'YData', NaN);
    end
end

function [V, F] = create_subdivided_sphere(level)
    t = (1.0 + sqrt(5.0)) / 2.0;
    V = [-1, t, 0; 1, t, 0; -1, -t, 0; 1, -t, 0; 0, -1, t; 0, 1, t; 0, -1, -t; 0, 1, -t; t, 0, -1; t, 0, 1; -t, 0, -1; -t, 0, 1];
    V = V ./ sqrt(sum(V.^2, 2));
    F = [1,12,6; 1,6,2; 1,2,8; 1,8,11; 1,11,12; 2,6,10; 6,12,5; 12,11,3; 11,8,7; 8,2,9; 4,10,5; 4,5,3; 4,3,7; 4,7,9; 4,9,10; 5,10,6; 3,5,12; 7,3,11; 9,7,8; 10,9,2];
    for i = 1:level
        edges = [F(:,1), F(:,2); F(:,2), F(:,3); F(:,3), F(:,1)];
        [u_edges, ~, ic] = unique(sort(edges, 2), 'rows');
        midpoints = (V(u_edges(:,1), :) + V(u_edges(:,2), :)) / 2;
        midpoints = midpoints ./ sqrt(sum(midpoints.^2, 2));
        num_V_old = size(V, 1); V = [V; midpoints];
        m1 = num_V_old + ic(1:size(F,1)); m2 = num_V_old + ic(size(F,1)+1:2*size(F,1)); m3 = num_V_old + ic(2*size(F,1)+1:end);
        F = [F(:,1), m1, m3; F(:,2), m2, m1; F(:,3), m3, m2; m1, m2, m3];
    end
end

function vol = compute_volume(V, F)
    v1 = V(F(:,1), :); v2 = V(F(:,2), :); v3 = V(F(:,3), :);
    vol = abs(sum(dot(v1, cross(v2, v3, 2), 2)) / 6);
end

function area = compute_area(V, F)
    v1 = V(F(:,1), :); v2 = V(F(:,2), :); v3 = V(F(:,3), :);
    area = sum(0.5 * sqrt(sum(cross(v2 - v1, v3 - v1, 2).^2, 2)));
end

function setup_2d_axes(ax, title_str, x_lim, y_lim)
    xlim(ax, x_lim); ylim(ax, y_lim);
    title(ax, title_str, 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel(ax, 'Cycles', 'FontName', 'Times New Roman', 'FontSize', 10);
    grid(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 10);
end