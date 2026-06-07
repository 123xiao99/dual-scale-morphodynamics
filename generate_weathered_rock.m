function generate_abraded_particle_curvature_with_data()
    % ==========================================
    % 颗粒地质磨蚀静态对比 (水平排版 + 蓝灰红色阶 + 完整数据标题)
    % ==========================================
    
    % --- 1. 核心动力学与衰减参数User-defined parameters ---
    num_faces = 25;              
    total_cycles = 0;          
    
    abrasion_rate = 0.002;        
    D9_15_start = 0.005;         
    roughness_decay_rate = 0.002;
    
    scale_xyz = [1.0, 1.0, 1.0]; 
    mesh_level = 4;              
    random_seed = 2026;          
    
    rng(random_seed);
    fprintf('1. 正在初始化基础原石模型...\n');
    
    % --- 2. 构造非对称 Voronoi 原石 ---
    phi_pts = acos(2 * rand(num_faces, 1) - 1);
    theta_pts = 2 * pi * rand(num_faces, 1);
    x_pts = 2 * sin(phi_pts) .* cos(theta_pts);
    y_pts = 2 * sin(phi_pts) .* sin(theta_pts);
    z_pts = 2 * cos(phi_pts);
    
    pts = [x_pts, y_pts, z_pts; 0, 0, 0]; 
    [V_vor, C] = voronoin(pts);
    v_cell = V_vor(C{end}, :); 
    
    v_cell = v_cell .* scale_xyz;
    v_cell = v_cell - mean(v_cell, 1);          
    v_cell = v_cell / mean(sqrt(sum(v_cell.^2, 2))); 
    K = convhull(v_cell);
    
    % --- 3. 生成均匀球面网格并映射 ---
    [V_sphere, F] = create_subdivided_sphere(mesh_level);
    r_base = inf(size(V_sphere, 1), 1);
    Rx = V_sphere(:,1); Ry = V_sphere(:,2); Rz = V_sphere(:,3);
    for i = 1:size(K, 1)
        A = v_cell(K(i,1), :); B = v_cell(K(i,2), :); C_vert = v_cell(K(i,3), :);
        N = cross(B-A, C_vert-A); N = N / norm(N);
        d_plane = dot(A, N);
        if d_plane < 0, N = -N; d_plane = -d_plane; end
        dot_RN = Rx .* N(1) + Ry .* N(2) + Rz .* N(3);
        t = d_plane ./ dot_RN; t(dot_RN <= 0) = inf;
        r_base = min(r_base, t);
    end
    V_base = V_sphere .* r_base; 
    
    % --- 4. 构造 3D 拉普拉斯算子与均匀微观纹理场 ---
    fprintf('2. 正在构建 3D 拉普拉斯算子与空间微观纹理...\n');
    E1 = [F(:,1); F(:,2); F(:,3)]; E2 = [F(:,2); F(:,3); F(:,1)];
    A_mat = sparse(E1, E2, 1, size(V_sphere,1), size(V_sphere,1));
    A_mat = A_mat | A_mat'; 
    invD = spdiags(1./sum(A_mat,2), 0, size(V_sphere,1), size(V_sphere,1));
    L_op = invD * A_mat; 
    
    dx_sh = zeros(size(V_sphere, 1), 1); dy_sh = zeros(size(V_sphere, 1), 1); dz_sh = zeros(size(V_sphere, 1), 1);
    num_waves = 80; base_frequency = 15; 
    for w = 1:num_waves
        k_dir = randn(1, 3); k_dir = k_dir / norm(k_dir);
        freq = base_frequency + rand() * 8; phase = rand() * 2 * pi;
        wave_val = sin(freq * (V_base * k_dir') + phase); 
        amp_dir = randn(1, 3); amp_dir = amp_dir / norm(amp_dir);
        dx_sh = dx_sh + amp_dir(1) * wave_val; dy_sh = dy_sh + amp_dir(2) * wave_val; dz_sh = dz_sh + amp_dir(3) * wave_val;
    end
    noise_magnitude = sqrt(dx_sh.^2 + dy_sh.^2 + dz_sh.^2);
    V_roughness_unit = [dx_sh, dy_sh, dz_sh] / mean(noise_magnitude);
    
    % --- 5. 正向动力学演化推演 ---
    fprintf('3. 正在执行宏观去棱角与微观抛光耦合计算...\n');
    
    V1 = V_base;                                         
    vol1 = compute_volume(V1, F);
    
    V2 = V_base + V_roughness_unit * D9_15_start;        
    vol2 = compute_volume(V2, F);
    
    V_current = V_base;
    for t = 1:total_cycles
        V_avg = L_op * V_current;
        V_current = (1 - abrasion_rate) * V_current + abrasion_rate * V_avg;
    end
    final_D9_15 = D9_15_start * exp(-roughness_decay_rate * total_cycles);
    V3 = V_current + V_roughness_unit * final_D9_15;       
    vol3 = compute_volume(V3, F);
    
    % --- 计算曲率 ---
    fprintf('4. 正在计算表面凹凸曲率场...\n');
    laplacian_vec = V3 - L_op * V3; 
    normals = V3 ./ sqrt(sum(V3.^2, 2)); 
    curvature = sum(laplacian_vec .* normals, 2); 
    
    % --- 6. 绘图与渲染 ---
    fprintf('5. 正在生成排版图像...\n');
    hFig = figure('Color', 'w', 'Name', 'Particle Evolution (Curvature Map with Data)', 'Position', [100, 300, 1600, 450]);
    
    color_base = [0.65 0.65 0.65]; 
    color_sh   = [0.8 0.4 0.4]; 
    alpha_val  = 0.95;           
    
    % 图 1 (左)
    ax1 = subplot(1, 3, 1);
    trisurf(F, V1(:,1), V1(:,2), V1(:,3), 'FaceColor', color_base, 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
    setup_lighting_view(ax1, sprintf('Initial Base (15-Face)\nVolume: %.3f', vol1));
    
    % 图 2 (中)
    ax2 = subplot(1, 3, 2);
    trisurf(F, V2(:,1), V2(:,2), V2(:,3), 'FaceColor', color_sh, 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
    setup_lighting_view(ax2, sprintf('Angular (D_{9-15} = %g)\nVolume: %.3f', D9_15_start, vol2));
    
    % 图 3 (右) - 曲率渲染 + 完整数据
    ax3 = subplot(1, 3, 3);
    trisurf(F, V3(:,1), V3(:,2), V3(:,3), curvature, 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    
    material(ax3, 'dull'); camlight(ax3, 'headlight'); camlight(ax3, 'left'); 
    lighting(ax3, 'gouraud'); shading(ax3, 'interp'); 
    axis(ax3, 'equal', 'off'); grid(ax3, 'off'); view(ax3, [30, 30]);
    
    set(ax3, 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.05);
    xlabel(ax3, 'X-axis', 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.05, 'FontWeight', 'bold'); 
    ylabel(ax3, 'Y-axis', 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.05, 'FontWeight', 'bold'); 
    zlabel(ax3, 'Z-axis', 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.05, 'FontWeight', 'bold');
    
    % ★ 核心修改：恢复并丰富第三个子图的量化数据标题 ★
    vol_ratio = (vol3 / vol2) * 100;
    vol_loss  = 100 - vol_ratio;
    title_str3 = sprintf('Abraded: Curvature (D_{9-15} = %.4f)\nVol Ratio: %.1f%% | Loss: %.1f%%', final_D9_15, vol_ratio, vol_loss);
    title(ax3, title_str3, 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.06, 'FontWeight', 'bold');
    
    % 应用新的蓝-灰-红色阶
    colormap(ax3, create_blue_grey_red_cmap());
    caxis(ax3, [-0.03, 0.03]);       
    
    % 修复的 ColorBar 
    cb = colorbar(ax3, 'eastoutside');
    set(cb, 'FontName', 'Times New Roman', 'FontSize', 11);
    title(cb, 'Curvature', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');
    
    hlink = linkprop([ax1, ax2, ax3], {'CameraPosition', 'CameraUpVector', 'CameraTarget', 'CameraViewAngle', 'XLim', 'YLim', 'ZLim'});
    setappdata(hFig, 'StoreTheLink', hlink); 
    
    fprintf('完成！体积比值、损失率及最终的 D9-15 已精确显示在第三幅图上方。\n');
end

% === 子函数：生成 3D 测地网格 ===
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

% === 子函数：计算 3D 体积 ===
function vol = compute_volume(V, F)
    v1 = V(F(:,1), :); v2 = V(F(:,2), :); v3 = V(F(:,3), :);
    vol = abs(sum(dot(v1, cross(v2, v3, 2), 2)) / 6);
end

% === 子函数：动态缩放与视图控制 (用于图1和图2) ===
function setup_lighting_view(ax, title_str)
    material(ax, 'dull'); camlight(ax, 'headlight'); camlight(ax, 'left'); lighting(ax, 'gouraud'); 
    axis(ax, 'equal', 'on'); grid(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.05);
    xlabel(ax, 'X-axis', 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.05, 'FontWeight', 'bold'); 
    ylabel(ax, 'Y-axis', 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.05, 'FontWeight', 'bold'); 
    zlabel(ax, 'Z-axis', 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.05, 'FontWeight', 'bold');
    view(ax, [30, 30]); 
    title(ax, title_str, 'FontName', 'Times New Roman', 'FontUnits', 'normalized', 'FontSize', 0.06, 'FontWeight', 'bold');
end

% === 子函数：生成深蓝-岩石灰-深红 专属色阶 ===
function cmap = create_blue_grey_red_cmap()
    n = 128;
    r1 = linspace(0.1, 0.65, n)'; g1 = linspace(0.3, 0.65, n)'; b1 = linspace(0.8, 0.65, n)';
    r2 = linspace(0.65, 0.8, n)'; g2 = linspace(0.65, 0.2, n)'; b2 = linspace(0.65, 0.2, n)';
    cmap = [r1, g1, b1; r2(2:end), g2(2:end), b2(2:end)];
end