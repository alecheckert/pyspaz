function [ trackedPar ] = convert_trajs_to_mat()
% Convert Python trajectories files to MATLAB format

directory_name = pwd();
file_list = dir(directory_name);
[n_files, throw_away] = size(file_list);
for f_idx=1:n_files
    if strfind(file_list(f_idx).name, 'Tracked.mat')
        load(file_list(f_idx).name);
        [n_trajs, throw_away] = size(trackedPar);
        old_TP = trackedPar;
        clear trackedPar;
        trackedPar(1, n_trajs) = struct();
        for traj_idx=1:n_trajs
            trackedPar(traj_idx).xy = old_TP{traj_idx,1};
            trackedPar(traj_idx).Frame = transpose(old_TP{traj_idx,2});
            trackedPar(traj_idx).TimeStamp = transpose(old_TP{traj_idx,3});
        end
        new_name = strrep(file_list(f_idx).name, 'Tracked.mat', 'matlab-style_Tracked.mat');
        save(new_name, 'trackedPar');
    end
end

end
