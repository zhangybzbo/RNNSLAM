name=~/projects/ColonData/sequences/
num=031
python refining/my_system.py \
	--cameras_file_path $name$num'/reproduce/kf_pose_result.txt' \
	--depth_dir $name$num'/reproduce/depth' \
	--image_dir $name$num'/image/' \
	--intrinsic $name'calib_270_216.txt' \
	--presort_poses \
	--refined_depth_dir $name$num'/reproduce/depth_refined_145000' \
	--local_window_size 7  --use_viewer
