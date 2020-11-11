name="/home/zhangyb/projects/ColonData/sequences/"
num="038"
python python/windowed_depth_average/main.py \
        --cameras_file_path "$name$num/reproduce/kf_pose_result.txt" \
        --depth_dir "$name$num/reproduce/depth" \
        --image_dir "$name$num/img_corr/" \
        --intrinsic ~/projects/ColonData/sequences/calib_270_216.txt \
        --presort_poses \
        --refined_depth_dir "$name$num/reproduce/depth_refined_145000" \
        --local_window_size 5  --use_viewer
